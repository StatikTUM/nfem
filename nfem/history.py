import copy

class History(object):

    def __init__(self, model):
        self.history = []
        self.AddModel(0, model)

    def GetModel(self, step):
        return self.history[step]

    def AddModel(self, step, model):
        if len(self.history) != step:
            raise RuntimeError("History has the wrong size!")
        self.history.append(copy.deepcopy(model))

    def ResetHistoryToStep(self, step):
        self.history = self.history[0:step]
        return self.GetModel(step)
    
    def ReturnHistorySize(self):
        return len(self.history)
