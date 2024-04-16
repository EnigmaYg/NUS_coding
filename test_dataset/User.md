### the Server Connection

for A5000*4

`ssh -p 20002 -J nextLabUser@gateway.ncl.sg zmyang@172.18.178.10`

for A40*4   

`ssh -p 5001 -J nextLabUser@gateway.ncl.sg zmyang@ncl-cr3.d2.comp.nus.edu.sg`

### SSH Files
#### take A40 as an example:

Downloading: scp -P 5001 -J nextLabUser@gateway.ncl.sg zmyang@ncl-cr3.d2.comp.nus.edu.sg:"path/to/server_file" "path/to/local"

`scp -P 5001 -J nextLabUser@gateway.ncl.sg zmyang@ncl-cr3.d2.comp.nus.edu.sg:/mnt/hdd/zmyang/EE/phase4/dataset_construction/results/level1/2022_events.csv /Users/milesyzm/Downloads/EventForecasting-main/phase4/dataset_construction`

Uploading: scp -P 5001 -J nextLabUser@gateway.ncl.sg "path/to/local_file" zmyang@ncl-cr3.d2.comp.nus.edu.sg:"path/to/server"

`scp -P 5001 -J nextLabUser@gateway.ncl.sg /Users/milesyzm/Documents/learning\(homework-notes-information\)/EF/MIDEAST/prompt1.py zmyang@ncl-cr3.d2.comp.nus.edu.sg:/mnt/hdd/zmyang/MIDEAST_new`

`scp -P 5001 -J nextLabUser@gateway.ncl.sg /Users/milesyzm/Downloads/EventForecasting-main/phase4/dataset_construction/fastchat/src_all/parse_level1.py /Users/milesyzm/Downloads/EventForecasting-main/phase4/dataset_construction/fastchat/src_all/parse_level2.py /Users/milesyzm/Downloads/EventForecasting-main/phase4/dataset_construction/fastchat/src_all/parse_level3.py zmyang@ncl-cr3.d2.comp.nus.edu.sg:/mnt/hdd/zmyang/EE/phase4/FastChat/fastchat/src_all`

### Execution

Load in vicuna-7b-v1.5 (use gpu1):

`CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.src_all.prompt_level1 --model /mnt/hdd/llm/vicuna_hf/vicuna-7b-v1.5 --load-8bit --year 2022 --output_path /mnt/hdd/zmyang/EE`

`CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.src_all.parse_level3 --year 2022 --output_path /mnt/hdd/zmyang/EE`

Load in vicuna-13b (use gpu0):

`CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.src_all.prompt_level1 --model /mnt/hdd/llm/vicuna-13b --load-8bit --year 2021 --output_path /mnt/hdd/zmyang/EE`

`CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.src_all.parse_level1 --year 2021 --output_path /mnt/hdd/zmyang/EE`

### Addition

Watch Loading

`watch -n 1 nvidia-smi`

monitor

`htop -u zmyang`

### Network

`ssh -L 127.0.0.1:8080:127.0.0.1:8888 -p 5001 -J nextLabUser@gateway.ncl.sg zmyang@ncl-cr3.d2.comp.nus.edu.sg`
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.src_all.prompt_che --model /mnt/hdd/llm/vicuna_hf/vicuna-7b-v1.5 --load-8bit --input_path /mnt/hdd/zmyang/CHEMPROT/ --output_path /mnt/hdd/zmyang/EE
