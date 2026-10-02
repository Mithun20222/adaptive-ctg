from metrics import contains


class AdaptiveRouter:
    """Decides, at step t, whether to hand a span to COLD."""

    def __init__(self, keywords, L, tau_horizon=5.0, extra=3):
        self.keywords = keywords          # list of keywords, each a list of token ids
        self.L = L                        # total target length
        self.tau = tau_horizon            # max tokens-per-remaining-keyword before escalating
        self.extra = extra                # span = len(keyword) + extra

    def remaining(self, generated):
        return [kw for kw in self.keywords if not contains(generated, kw)]

    def check(self, t, generated):
        """t = tokens generated so far; generated = their ids (list).
        Returns None (stay with BOLT) or (keyword, span_length) (escalate to COLD)."""
        rem = self.remaining(generated)
        if not rem:
            return None                                   # all constraints met
        delta = (self.L - t) / len(rem)                   # horizon deficit from the spec
        if delta > self.tau:
            return None                                   # enough room per keyword
        kw = rem[0]
        if self.L - t < len(kw):
            return None                                   # keyword cannot fit anymore
        return kw, min(len(kw) + self.extra, self.L - t)