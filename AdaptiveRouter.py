from metrics import contains


class AdaptiveRouter:
    """Decides, at step t, whether to hand a span to COLD."""

    def __init__(self, keywords, L, tau_horizon=5.0, extra=3, single=False):
        self.single = single
        self.keywords = keywords          # list of keywords, each a list of token ids
        self.L = L                        # total target length
        self.tau = tau_horizon            # max tokens per remaining keyword before escalating
        self.extra = extra                # slack tokens added to the span

    def remaining(self, generated):
        return [kw for kw in self.keywords if not contains(generated, kw)]

    def check(self, t, generated):
        """t = tokens generated so far; generated = their ids (list).
        Returns None (stay with BOLT) or (keywords, k): COLD span of k tokens
        covering every remaining keyword that still fits."""
        rem = self.remaining(generated)
        if not rem:
            return None                                   # all constraints met
        if (self.L - t) / len(rem) > self.tau:
            return None                                   # enough room per keyword
        room = self.L - t
        fit = [kw for kw in rem if len(kw) <= room]       # keywords that can still fit
        if not fit:
            return None
        if self.single:
            fit = fit[:1] 
        k = min(sum(len(kw) for kw in fit) + self.extra, room)
        return fit, k
    
    def check_suffix(self, t, bridge_len):
        """Returns k (span length for COLD) once the remaining room is within the bridge."""
        room = self.L - t
        return room if room <= bridge_len else None