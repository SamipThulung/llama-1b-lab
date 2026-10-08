# llama-1b-lab
A hands-on, modular repo to pretraining, scaling, and aligning a 1B Llama model from scratch.


48gb A40
NCCL_P2P_DISABLE=1 python -m ddp_05.naive_ddp_debug
DDPs run comfortable with m size model
but runs only to some steps in l size model 
loss after 100 epoch
s 1.9 loss 
m 2.01
l Step: 99, loss: 2.76701

FSDP:
TINY MODEL
ls = Step: 99, loss: 6.62315

Kill process
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f naive_ddp

Git sshkey
git remote set-url origin git@github.com:SamipThulung/llama-1b-lab.git
git config --global user.email "samipthulung3@gmail.com"
git config --global user.name "SamipThulung"
ssh-keygen -t ed25519 -C "samipthulung3@gmail.com"
cat ~/.ssh/id_ed25519.pub

git push origin main

Fetch GitHub's host key into known_hosts
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts

Verifying-finger print
ssh-keygen -lf ~/.ssh/known_hosts
Test Authentication
ssh -T git@github.com

echo "filename.txt" >> .gitignore

Initial run megatron
pip install -r requirements.txt
python -m dataset_03.megatron_data
torchrun --nproc_per_node=2 -m megatron_07.megatron_train
NCCL_P2P_DISABLE=1 python -m megatron_07.megatron_train # for l4 gpu
niteration   99 | lr 3.01e-05 | loss 1.8429 | grad_norm 0.5999701343494377 | step_ok True
n iteration   99 | lr 3.01e-05 | loss 1.8429 | grad_norm 0.5999701343494377 | step_ok True
s iteration   99 | lr 3.01e-05 | loss 1.7891 | grad_norm 0.7646029131164056 | step_ok True
s iteration  199 | lr 3.00e-05 | loss 1.7078 | grad_norm 0.7362759113771518 | step_ok True
iteration  999 | lr 3.00e-05 | loss 1.4359 | grad_norm 0.7253650929893452 | step_ok True

m iteration   99 | lr 3.01e-05 | loss 1.8072 | grad_norm 0.8433550864141988 | step_ok True
l iteration   99 | lr 3.01e-05 | loss 1.9120 | grad_norm 1.2634242668657127 | step_ok True
