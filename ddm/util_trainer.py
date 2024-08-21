import torch

class LossLogger:
    def __init__(self):
        self.loss_history=[]
        self.loss_dict_history=[]

    def start_epoch(self):
        self.running_loss = 0
        self.running_loss_dict = {}
        self.running_count = 0

    def update(self, loss, loss_dict):
        """
        Update batch loss

        This method should be called for each iteration during batch training 

        Arsgs:
          loss (tensor): scaler loss
          loss dict (Dict[str,tensor]): key=loss name value=scaler loss
        """
        self.running_loss += loss
        self.running_count +=1
        for k, v in loss_dict.items():
            if k in self.running_loss_dict:
                self.running_loss_dict[k] += v
            else:
                self.running_loss_dict[k] = v

    def end_epoch(self,mean_flag=True):
        """
        This method should be called after each epoch 

        Arsgs:
          mean_flag (bool): mean all losses computed in iterations
        """
        if mean_flag:
            self.running_loss /= self.running_count
            for k in self.running_loss_dict.keys():
                self.running_loss_dict[k] /=  self.running_count
        self.loss_history.append(self.running_loss)
        self.loss_dict_history.append(self.running_loss_dict)

    def get_dict(self, prefix="train"):
        """
        Arsgs:
          prefix (str): train/valid/test
        Returns:
          results Dict[str,float]:  key=loss name value=loss 
        """
        result={}
        key="{:s}-loss".format(prefix)
        val=self.running_loss
        result[key]=float(val)
        for k, v in self.running_loss_dict.items():
            if k[0]!="*":
                m = "{:s}-{:s}-loss".format(prefix, k)
            else:
                m = "*{:s}-{:s}".format(prefix, k[1:])
            result[m]=float(v)
        return result
    def get_msg(self, prefix="train"):
        """
        Arsgs:
          prefix (str): train/valid/test
        Returns:
          results Dict[str,float]:  key=loss name value=loss 
        """
        msg = []
        for key,val in self.get_dict(prefix=prefix).items():
            m = "{:s}: {:.3f}".format(key,val)
            msg.append(m)
        return "  ".join(msg)

    def get_loss(self):
        return self.running_loss
