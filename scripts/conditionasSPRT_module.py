#REFERENCE
# A Conditional Sequential Test for the Equality of Two Binomial Proportions
# William Q. Meeker, Jr
# Journal of the Royal Statistical Society. Series C (Applied Statistics)
# Vol. 30, No. 2 (1981), pp. 109-115
class ConditionalSPRT:
    def __init__(self, exposed, control, odd_ratio, alpha=0.05, beta=0.10, stop=None):
        self.exposed = exposed
        self.control = control
        self.odd_ratio = odd_ratio
        self.alpha = alpha
        self.beta = beta
        self.stop = stop

    def run(self):
        res = conditionalSPRT( self.exposed,
                               self.control,
                               self.odd_ratio,
                               self.alpha,
                               self.beta,
                               self.stop)
        return res

    def jsonResult(self, res):
        outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits = res
        res = {
            "decsionMade": outcome,
            "numberOfObservation": len(n),
            "truncated": truncated,
            "truncateDecision": truncate_decision
        }
        return res

    def plotExperiment(self, res):
        outcome,n, k,l,u,truncated,truncate_decision,x1,r,stats,limits = res
        lower = limits[:, 0]
        upper = limits[:,1]

        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(n, x1, label='Cumlative value of yes+no')

        ax.plot(n, lower, label='Lower Bound')
        ax.plot(n, upper, label='Upper Bound')

        plt.legend()
        plt.show()