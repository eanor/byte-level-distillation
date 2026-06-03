# byte-level-distillation
Cross-tokenizer knowledge distillation via byte-level alignment. Replaces the incorrect token merging formula p(y|x) · ∏ p(t̂ᵢ|t̂&lt;ᵢ, x) with a byte-space JSD loss where both teacher and student are conditioned on identical contexts.
