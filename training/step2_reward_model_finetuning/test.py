# from transformers import AutoTokenizer, AutoModelForMaskedLM
#
#
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num}
# if __name__ == '__main__':
#     model = AutoModelForMaskedLM.from_pretrained("facebook/mbart-large-50")
#     print(get_parameter_number(model))