from tester import Tester
import torch

def construct_test_hook(test_dataloader):
    def hook(trainer):
        model = trainer.model
        model = model.eval()
        t = Tester(model, test_dataloader)
        trainer.epoch_test_vectors, trainer.epoch_test_labels = t.test()
    return hook

def construct_metric_hook(metric_func):
    def hook(trainer):
        torch.cuda.empty_cache()
        trainer.epoch_val_acc =\
            metric_func(trainer.epoch_test_vectors,
                        trainer.epoch_test_labels)
    return hook
