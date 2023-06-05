from iesta.machine_learning.dataloader import IESTAData, LABELS
from iesta.machine_learning.huggingface_loader import IESTAHuggingFace

conservative_data_obj = IESTAData(ideology="conservative", keep_labels = LABELS.EFF_INEFF, )
conservative_hf = IESTAHuggingFace(conservative_data_obj)
conservative_w_effect = conservative_hf.upload_w_labels(is_for_style_classifier=False)
conservative_w_effect_for_sc = conservative_hf.upload_w_labels(is_for_style_classifier=True)

liberal_data_obj = IESTAData(ideology="liberal", keep_labels = LABELS.EFF_INEFF)
liberal_hf = IESTAHuggingFace(liberal_data_obj)
liberal_w_effect = liberal_hf.upload_w_labels(is_for_style_classifier=False, force_reload=True)
liberal_w_effect_for_sc = liberal_hf.upload_w_labels(is_for_style_classifier=True, force_reload=True)




