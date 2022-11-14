import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]

    
    def forward_cam(self, image, target, **kwargs):
        with torch.no_grad():
            logits_student, feature_student = self.student(image)
            logits_teacher, feature_teacher = self.teacher(image)
        pred_teacher = logits_teacher.softmax(1).max(1)

        # losses
        num_classes = self.fc_s.weight.shape[0]
        feature_student = feature_student["feats"][-1]
        feature_teacher = feature_teacher["feats"][-1]

        linear_student = self.fc_s.weight.view(num_classes, 1, -1, 1, 1)
        linear_teacher = self.fc_t.weight.view(num_classes, 1, -1, 1, 1)
        cam_student = ((feature_student) * linear_student).mean(2).clamp(0)
        cam_teacher = ((feature_teacher) * linear_teacher).mean(2).clamp(0)
        pdb.set_trace()
        losses_dict = {}
        return logits_student, losses_dict