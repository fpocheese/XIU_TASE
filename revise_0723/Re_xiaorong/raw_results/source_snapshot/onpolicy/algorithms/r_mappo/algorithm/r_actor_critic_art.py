"""Paper-auditable ART-MAPPO actor and critic.

This module is intentionally separate from the historical
``r_actor_critic_advanced.py`` implementation.  The latter applies attention
to a length-one sequence and therefore cannot model interactions between
physical observation channels.  Here every local physical channel is a token,
as specified in the manuscript, and the three reviewer-requested components
can be disabled independently.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.util import check, init
from onpolicy.utils.util import get_shape_from_obs_space


class ResidualBlock(nn.Module):
    """Two-layer pre-activation residual block used in the paper."""

    def __init__(self, hidden_size: int, use_orthogonal: bool) -> None:
        super().__init__()
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        self.fc1 = init_(nn.Linear(hidden_size, hidden_size))
        self.fc2 = init_(nn.Linear(hidden_size, hidden_size))
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch = F.relu(self.ln1(self.fc1(x)))
        branch = self.ln2(self.fc2(branch))
        return F.relu(x + branch)


class TaskAwareEncoder(nn.Module):
    """Attention-residual encoder with physically meaningful tokens.

    For a decentralized actor, ``token_width`` is one, hence each token is one
    of the 17 normalized physical channels.  For the centralized critic, each
    token is one defender's 17-dimensional observation; attention therefore
    captures cross-defender interactions without the quadratic cost of 340
    scalar tokens.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        token_width: int,
        embed_dim: int,
        num_heads: int,
        residual_blocks: int,
        use_backbone: bool,
        use_orthogonal: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim % token_width != 0:
            raise ValueError(
                f"input_dim={input_dim} must be divisible by token_width={token_width}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.input_dim = input_dim
        self.token_width = token_width
        self.num_tokens = input_dim // token_width
        self.hidden_size = hidden_size
        self.use_backbone = use_backbone

        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(module, gain=1.0):
            return init(
                module,
                init_method,
                lambda x: nn.init.constant_(x, 0),
                gain=gain,
            )

        if use_backbone:
            # Separate affine embedding f_r for every physical token.
            self.token_weight = nn.Parameter(
                torch.empty(self.num_tokens, token_width, embed_dim)
            )
            self.token_bias = nn.Parameter(torch.zeros(self.num_tokens, embed_dim))
            nn.init.xavier_uniform_(self.token_weight)
            self.attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_dropout = nn.Dropout(dropout)
            self.project = init_(
                nn.Linear(self.num_tokens * embed_dim, hidden_size)
            )
            self.residual = nn.ModuleList(
                ResidualBlock(hidden_size, use_orthogonal)
                for _ in range(residual_blocks)
            )
        else:
            # Capacity-controlled plain MLP replacement.  Its width is solved
            # from the parameter count of the removed token-attention,
            # projection, and residual stack, so the ablation tests topology
            # rather than merely reducing model size.
            target_params = (
                self.num_tokens * token_width * embed_dim
                + self.num_tokens * embed_dim
                + 4 * embed_dim * embed_dim
                + 4 * embed_dim
                + self.num_tokens * embed_dim * hidden_size
                + hidden_size
                + residual_blocks
                * (2 * hidden_size * hidden_size + 6 * hidden_size)
            )
            linear_term = input_dim + hidden_size + 2
            plain_width = max(
                hidden_size,
                int(
                    round(
                        0.5
                        * (
                            -linear_term
                            + math.sqrt(
                                linear_term * linear_term
                                + 4 * max(target_params - hidden_size, 1)
                            )
                        )
                    )
                ),
            )
            self.plain = nn.Sequential(
                init_(nn.Linear(input_dim, plain_width)),
                nn.ReLU(),
                init_(nn.Linear(plain_width, plain_width)),
                nn.ReLU(),
                init_(nn.Linear(plain_width, hidden_size)),
                nn.ReLU(),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_backbone:
            return self.plain(x)
        token_input = x.reshape(-1, self.num_tokens, self.token_width)
        tokens = torch.einsum("bnt,nte->bne", token_input, self.token_weight)
        tokens = tokens + self.token_bias.unsqueeze(0)
        attended, _ = self.attention(
            tokens, tokens, tokens, need_weights=False
        )
        tokens = tokens + self.attention_dropout(attended)
        features = F.relu(self.project(tokens.flatten(start_dim=1)))
        for block in self.residual:
            features = block(features)
        return features


class R_Actor_ART(nn.Module):
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_gru = bool(getattr(args, "art_use_gru", True))
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) != 1:
            raise ValueError("ART-MAPPO expects a flat physical observation vector")
        obs_dim = int(obs_shape[0])
        self.encoder = TaskAwareEncoder(
            input_dim=obs_dim,
            hidden_size=self.hidden_size,
            token_width=1,
            embed_dim=int(getattr(args, "art_attention_embed_dim", 16)),
            num_heads=int(getattr(args, "art_attention_heads", 4)),
            residual_blocks=int(getattr(args, "art_residual_blocks", 2)),
            use_backbone=bool(getattr(args, "art_use_attention_residual", True)),
            use_orthogonal=self._use_orthogonal,
            dropout=float(getattr(args, "art_attention_dropout", 0.0)),
        )
        if self._use_gru:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )
        self.act = ACTLayer(
            action_space,
            self.hidden_size,
            self._use_orthogonal,
            self._gain,
        )
        self.to(device)

    def _features(self, obs, rnn_states, masks):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        features = self.encoder(obs)
        if self._use_gru:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return features, rnn_states

    def forward(
        self,
        obs,
        rnn_states,
        masks,
        available_actions=None,
        deterministic=False,
    ):
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        features, rnn_states = self._features(obs, rnn_states, masks)
        actions, action_log_probs = self.act(
            features, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states

    def evaluate_actions(
        self,
        obs,
        rnn_states,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        action = check(action).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
        features, _ = self._features(obs, rnn_states, masks)
        return self.act.evaluate_actions(
            features,
            action,
            available_actions,
            active_masks=(
                active_masks if self._use_policy_active_masks else None
            ),
        )


class R_Critic_ART(nn.Module):
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_gru = bool(getattr(args, "art_use_gru", True))
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_shape = get_shape_from_obs_space(cent_obs_space)
        if len(cent_shape) != 1:
            raise ValueError("ART-MAPPO expects a flat centralized state")
        cent_dim = int(cent_shape[0])
        local_obs_dim = int(getattr(args, "art_local_obs_dim", 17))
        token_width = local_obs_dim if cent_dim % local_obs_dim == 0 else 1
        self.encoder = TaskAwareEncoder(
            input_dim=cent_dim,
            hidden_size=self.hidden_size,
            token_width=token_width,
            embed_dim=int(getattr(args, "art_attention_embed_dim", 16)),
            num_heads=int(getattr(args, "art_attention_heads", 4)),
            residual_blocks=int(getattr(args, "art_residual_blocks", 2)),
            use_backbone=bool(getattr(args, "art_use_attention_residual", True)),
            use_orthogonal=self._use_orthogonal,
            dropout=float(getattr(args, "art_attention_dropout", 0.0)),
        )
        if self._use_gru:
            self.rnn = RNNLayer(
                self.hidden_size,
                self.hidden_size,
                self._recurrent_N,
                self._use_orthogonal,
            )

        init_method = (
            nn.init.orthogonal_
            if self._use_orthogonal
            else nn.init.xavier_uniform_
        )

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        features = self.encoder(cent_obs)
        if self._use_gru:
            features, rnn_states = self.rnn(features, rnn_states, masks)
        return self.v_out(features), rnn_states
