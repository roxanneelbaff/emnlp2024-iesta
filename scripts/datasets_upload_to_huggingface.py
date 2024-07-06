from iesta.data.iesta_data import IESTAData, LABELS
from iesta.data.huggingface_loader import IESTAHuggingFace

import argparse


def upload(ideology, args):
    data_obj = IESTAData(
        ideology=ideology,
        keep_labels=LABELS.EFF_INEFF,
    )
    hf = IESTAHuggingFace(data_obj, reload_preprocess=args.reload_preprocess)
    if args.is_for_style_classifier in ["yes", "all"]:
        _ = hf.upload_w_labels(
            is_for_style_classifier=True, force_reload=args.force_reload
        )
    if args.is_for_style_classifier in ["no", "all"]:
        _ = hf.upload_w_labels(
            is_for_style_classifier=False, force_reload=args.force_reload
        )


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-s", "--is_for_style_classifier", type=str, default="all"
        )
        parser.add_argument(
            "-f", "--force_reload", action="store_true"
        )  # reload from and to hf
        parser.add_argument("-i", "--ideology", type=str)
        parser.add_argument("-r", "--reload_preprocess", action="store_true")
        args = parser.parse_args()
        print(args)
        if args.ideology == "conservative" or args.ideology == "all":
            upload("conservative", args)
        if args.ideology == "liberal" or args.ideology == "all":
            upload("liberal", args)
    finally:
        pass


if __name__ == "__main__":
    main()


# nohup python3 scripts/upload_datasets_to_huggingface.py -i all -f -s all  > logs/upload_to_huggingface.log  2>&1 &

##from iesta.data.finetuning_data import FineTuningData as FTD
# liberal = FTD.get_sv_data("liberal", skip_system_prompt=True, save=True)
