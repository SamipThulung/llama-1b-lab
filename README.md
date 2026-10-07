# llama-1b-lab
A hands-on, modular repo to pretraining, scaling, and aligning a 1B Llama model from scratch.


48gb A40
NCCL_P2P_DISABLE=1 python -m ddp_05.naive_ddp_debug
DDPs run comfortable with m size model
but runs only to some steps in l size model 

Kill process
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f naive_ddp

Git sshkey
git remote set-url origin git@github.com:SamipThulung/llama-1b-lab.git
ssh-keygen -t ed25519 -C "samipthulung3@gmail.com"
cat ~/.ssh/id_ed25519.pub
git config --global user.email "samipthulung3@gmail.com"
git config --global user.name "SamipThulung"
git push origin main
