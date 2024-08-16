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
        if isinstance(logits_student, list):
            loss = []
            # in each branch, less than one class > 0.5
            for br in logits_student:
                loss.append(br.sigmoid().topk(2)[0][:, 1].mean()) # len(loss)==3
            # in all branches, at least one class --> 1
            all_logits = torch.cat(logits_student, 1).sigmoid()
            loss.append(1 - all_logits.max(1)[0].mean()) # len(loss) == 4
            # for different classes, inner product --> 0
            class_diff = (target.unsqueeze(0) == target.unsqueeze(1)).logical_not()
            loss.append((all_logits @ all_logits.T)[class_diff].mean() * 2)
            # ce loss weighted 0.1
            # ce_loss = []
            for br in logits_student:
                # ce_loss.append(1 * F.cross_entropy(br, target))
                loss.append(1 * F.cross_entropy(br, target))
            # import pdb
            # pdb.set_trace()
            # all_br = torch.stack(logits_student, 2).max(2)[0]
            # loss.append(F.cross_entropy(all_br, target))
            # loss.append(min(ce_loss))
            # loss.append(ce_loss[-1])
            # assert(len(loss) == 8)
            
            loss = sum(loss)
            # logits_student_out = logits_student[2]
            # logits_student_out = all_br
            if logits_student[0].max() > 0:
                logits_student_out = logits_student[0]
            elif logits_student[1].max() > 0:
                logits_student_out = logits_student[1]
            else:
                logits_student_out = logits_student[2]
            return logits_student_out, {"ce": loss}

            # loss = F.cross_entropy(logits_student[2], target)
            # return logits_student[2], {"ce": loss}
        else:
            loss = F.cross_entropy(logits_student, target)
            # import pdb
            # pdb.set_trace()
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
        # pdb.set_trace()
        losses_dict = {}
        return logits_student, losses_dict