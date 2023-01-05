
'''
Class to summarize the architecture of a network. Could be useful info for cellular automata to have??
Not currently in use.
'''
class Summary:
    def __init__(self, model):
        self.depth = 0
        self.total_params = 0
        self.layer_sizes = []
        self.avg_relative_layer_size = 0  # Avg size of layer compared to first layer
        self.model_summary(model)

    def model_summary(self, model):
        model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
        layer_name = [child for child in model.children()]
        self.depth = len(layer_name)
        j = 0
        total_params = 0
        print("\t"*10)
        for i in layer_name:
            try:
                bias = (i.bias is not None)
            except:
                bias = False
            if not bias:
                param = model_parameters[j].numel()+model_parameters[j+1].numel()
                j = j+2
            else:
                param = model_parameters[j].numel()
                j = j+1
            # print(str(i)+"\t"*3+str(param))
            self.layer_sizes.append(param)

            total_params+=param
        self.total_params = total_params
        self.avg_relative_layer_size = np.average(self.layer_sizes) / self.layer_sizes[0]

    #XXX todo maybe would make sense to move Cell's architectureTo.. funcs into this
