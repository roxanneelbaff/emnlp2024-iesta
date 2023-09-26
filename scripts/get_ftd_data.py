from iesta.data.finetuning_data import FineTuningData as FTD
# 

def main():
    try:
        liberal = FTD.get_sv_data("liberal", skip_system_prompt=True, save=True)
        liberal = FTD.get_sv_data("liberal", skip_system_prompt=False, save=True)
        conservative = FTD.get_sv_data("conservative", skip_system_prompt=True, save=True)
        conservative = FTD.get_sv_data("conservative", skip_system_prompt=False, save=True)
    finally:
        pass


if __name__ == "__main__":
    main()


# nohup python3 scripts/upload_datasets_to_huggingface.py -i all -f -s all  > logs/upload_to_huggingface.log  2>&1 &

##from iesta.data.finetuning_data import FineTuningData as FTD
# liberal = FTD.get_sv_data("liberal", skip_system_prompt=True, save=True)