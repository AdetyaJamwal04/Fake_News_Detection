# 🚀 AWS EC2 Deployment Guide - Complete Step-by-Step

Deploy your Fake News Detector to AWS EC2 from scratch.

---

## 📋 Prerequisites

- AWS Account (free tier eligible)
- Credit/debit card for AWS verification
- Basic command line knowledge
- Your GitHub repository URL

---

## 🎯 Part 1: AWS Account Setup

### Step 1: Create AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Enter email and account name
4. Verify email
5. Enter payment information (won't be charged if staying in free tier)
6. Verify identity (phone verification)
7. Choose "Basic Support - Free"

**Time: 10-15 minutes**

---

## 💻 Part 2: Launch EC2 Instance

### Step 1: Access EC2 Dashboard

1. Login to AWS Console
2. Search for "EC2" in top search bar
3. Click "EC2" (Virtual Servers in the Cloud)
4. Select your region (top-right dropdown)
   - **Recommended:** `us-east-1` (N. Virginia) - cheapest
   - Or choose closest to your users

### Step 2: Launch Instance

1. Click **"Launch Instance"** (orange button)

2. **Name your instance:**
   ```
   Name: fake-news-detector
   ```

3. **Choose AMI (Operating System):**
   - Select: **Ubuntu Server 22.04 LTS**
   - Architecture: **64-bit (x86)**
   - ✅ Free tier eligible

4. **Choose Instance Type:**
   - Select: **t2.medium** (4 GB RAM)
   - Why? Your app needs ~2-3 GB for models
   - Note: t2.medium is NOT free tier (costs ~$33/month)
   - For free tier: t2.micro (but will be slow/crash)

5. **Key Pair (Important!):**
   - Click "Create new key pair"
   - Name: `fake-news-detector-key`
   - Type: RSA
   - Format: `.pem` (for Mac/Linux) or `.ppk` (for Windows/PuTTY)
   - Click "Create key pair"
   - **SAVE THE FILE!** You can't download it again

6. **Network Settings:**
   - Click "Edit"
   - **Security Group Name:** `fake-news-detector-sg`
   - **Rules to add:**
     ```
     ✅ SSH (Port 22) - Your IP (for security)
     ✅ HTTP (Port 80) - Anywhere (0.0.0.0/0)
     ✅ HTTPS (Port 443) - Anywhere (0.0.0.0/0)
     ✅ Custom TCP (Port 5000) - Anywhere (0.0.0.0/0) [for testing]
     ```
   
   Click "Add security group rule" for each

7. **Storage:**
   - **20 GB** gp3 (models need space)
   - Free tier: 30 GB (you're within limits)

8. **Advanced Details:**
   - Leave defaults

9. **Click "Launch Instance"** (orange button)

Wait 2-3 minutes for instance to start.

**Time: 5 minutes**

---

## 🔗 Part 3: Connect to Your Server

### Step 1: Get Instance Details

1. Click on your instance ID
2. Note the **Public IPv4 address** (e.g., `3.84.123.45`)
3. Wait for "Instance State" = **Running**
4. Wait for "Status Check" = **2/2 checks passed**

### Step 2: Connect via SSH

**For Mac/Linux:**
```bash
# Move key to safe location
mkdir -p ~/.ssh
mv ~/Downloads/fake-news-detector-key.pem ~/.ssh/
chmod 400 ~/.ssh/fake-news-detector-key.pem

# Connect (replace IP with yours)
ssh -i ~/.ssh/fake-news-detector-key.pem ubuntu@YOUR_EC2_IP
```

**For Windows (PowerShell):**
```powershell
# Connect (replace IP)
ssh -i C:\Users\YourName\Downloads\fake-news-detector-key.pem ubuntu@YOUR_EC2_IP
```

**For Windows (PuTTY):**
1. Open PuTTY
2. Host: `ubuntu@YOUR_EC2_IP`
3. Connection > SSH > Auth > Browse for `.ppk` file
4. Click "Open"

You should see Ubuntu welcome message!

**Time: 2 minutes**

---

## 🛠️ Part 4: Install Dependencies on Server

### Step 1: Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 2: Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker ubuntu

# Apply group changes
newgrp docker

# Verify
docker --version
# Should show: Docker version 24.x.x
```

### Step 3: Install Docker Compose

```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable
sudo chmod +x /usr/local/bin/docker-compose

# Verify
docker-compose --version
# Should show: Docker Compose version v2.x.x
```

**Time: 5-10 minutes**

---

## 📦 Part 5: Deploy Your Application

### Step 1: Clone Your Repository

```bash
# Install git if needed
sudo apt install git -y

# Clone your repo (replace with your GitHub URL)
git clone https://github.com/YOUR_USERNAME/Fake_News_Detector.git
cd Fake_News_Detector
```

### Step 2: Build Docker Image

```bash
# Build (takes 5-10 minutes first time)
docker build -t fake-news-detector .
```

**What's happening:**
- Installing Python packages
- Downloading SBERT model (~500 MB)
- Downloading BART model (~1.6 GB)
- Downloading Spacy model
- This is ONE-TIME - models cached in image

**Time: 10-15 minutes**

### Step 3: Run the Container

```bash
# Run the application
docker run -d \
  --name fake-news-api \
  -p 80:5000 \
  --restart unless-stopped \
  fake-news-detector
```

**Explanation:**
- `-d`: Run in background (detached)
- `--name`: Container name
- `-p 80:5000`: Map port 80 (public) to 5000 (container)
- `--restart`: Auto-restart if crash/reboot
- `fake-news-detector`: Your image name

### Step 4: Verify It's Running

```bash
# Check container status
docker ps

# Should show:
# CONTAINER ID   IMAGE                  STATUS
# abc123...      fake-news-detector     Up 2 minutes

# Check logs
docker logs fake-news-api

# Should see: "Running on http://0.0.0.0:5000"
```

**Time: 2 minutes**

---

## 🌐 Part 6: Test Your API

### From Your Local Computer:

```bash
# Health check (replace with your EC2 IP)
curl http://YOUR_EC2_IP/api/health

# Should return:
# {"status":"healthy"}

# Test fact-check
curl -X POST http://YOUR_EC2_IP/api/check \
  -H "Content-Type: application/json" \
  -d '{"claim": "Python is a programming language", "max_results": 2}'
```

**Or open in browser:**
```
http://YOUR_EC2_IP/
http://YOUR_EC2_IP/api/health
```

✅ **Your API is now live!**

---

## 🔐 Part 7: Set Up Domain Name (Optional)

### Option A: Use AWS Route 53

1. **Buy domain in Route 53:**
   - Go to Route 53 in AWS Console
   - Register domain (~$12/year for .com)

2. **Create A Record:**
   - Go to Hosted Zones
   - Click your domain
   - Create Record:
     - Type: A
     - Name: api
     - Value: Your EC2 IP
     - TTL: 300

3. **Access via domain:**
   ```
   http://api.yourdomain.com/api/health
   ```

### Option B: Use Freenom (Free)

1. Go to [freenom.com](https://freenom.com)
2. Register free domain (.tk, .ml, .ga)
3. Add DNS record:
   - Type: A
   - Host: api
   - IP: Your EC2 IP

**Time: 10-20 minutes**

---

## 🔒 Part 8: Set Up HTTPS (SSL Certificate)

### Install Certbot & Nginx

```bash
# Install Nginx
sudo apt install nginx -y

# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/factchecker
```

**Add this configuration:**
```nginx
server {
    listen 80;
    server_name api.yourdomain.com;  # Your domain

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 120s;
        proxy_read_timeout 120s;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/factchecker /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Get SSL Certificate:**
```bash
sudo certbot --nginx -d api.yourdomain.com
```

Follow prompts:
- Enter email
- Agree to terms
- Choose: Redirect HTTP to HTTPS

✅ **Now accessible via HTTPS!**
```
https://api.yourdomain.com/api/health
```

**Time: 15 minutes**

---

## 📊 Part 9: Monitoring & Maintenance

### Check Application Status

```bash
# Container status
docker ps

# View logs (last 50 lines)
docker logs --tail 50 fake-news-api

# Follow logs live
docker logs -f fake-news-api

# Check resource usage
docker stats fake-news-api
```

### Restart Application

```bash
# Restart container
docker restart fake-news-api

# Or rebuild and restart
docker stop fake-news-api
docker rm fake-news-api
docker build -t fake-news-detector .
docker run -d --name fake-news-api -p 80:5000 --restart unless-stopped fake-news-detector
```

### Update Code

```bash
# SSH to server
cd Fake_News_Detector

# Pull latest changes
git pull

# Rebuild and restart
docker stop fake-news-api
docker rm fake-news-api
docker build -t fake-news-detector .
docker run -d --name fake-news-api -p 80:5000 --restart unless-stopped fake-news-detector
```

---

## 💰 Part 10: Cost Management

### Current Setup Costs (Monthly):

```
EC2 t2.medium: ~$33/month
Data transfer: ~$1-5/month (first 15 GB free)
Storage (20 GB): ~$2/month
---
Total: ~$36-40/month
```

### Ways to Reduce Costs:

1. **Use t2.micro** (free tier)
   - Free for 12 months
   - Only 1 GB RAM (app will be slow/unstable)

2. **Stop instance when not in use:**
   ```bash
   # From AWS Console: Actions > Instance State > Stop
   # Only pay for storage (~$2/month)
   ```

3. **Use reserved instances:**
   - Commit to 1-3 years
   - Save up to 72%

4. **Set up billing alerts:**
   - AWS Console > Billing > Budgets
   - Create alert for $10, $20, $30

---

## 🔧 Troubleshooting

### Issue: Can't connect to EC2

**Solution:**
```bash
# Check security group allows your IP
# AWS Console > EC2 > Security Groups
# Edit inbound rules > Add your current IP to SSH rule
```

### Issue: Docker container exits immediately

**Solution:**
```bash
# Check logs
docker logs fake-news-api

# Common issues:
# - Port 5000 already in use
# - Out of memory (upgrade to t2.medium)
```

### Issue: API timeouts

**Solution:**
```bash
# Increase timeout in Nginx
sudo nano /etc/nginx/sites-available/factchecker

# Add to location block:
proxy_connect_timeout 180s;
proxy_read_timeout 180s;

sudo systemctl restart nginx
```

### Issue: Out of memory

**Solution:**
```bash
# Check memory
free -h

# Add swap memory
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## ✅ Deployment Checklist

- [ ] EC2 instance launched and running
- [ ] Security group configured correctly
- [ ] Successfully SSHed to server
- [ ] Docker and Docker Compose installed
- [ ] Repository cloned
- [ ] Docker image built successfully
- [ ] Container running (`docker ps` shows it)
- [ ] API responds to health check
- [ ] Can fact-check a claim successfully
- [ ] (Optional) Domain name configured
- [ ] (Optional) HTTPS/SSL certificate installed
- [ ] Monitoring set up
- [ ] Billing alerts configured

---

## 📚 Quick Reference Commands

```bash
# SSH to server
ssh -i ~/.ssh/fake-news-detector-key.pem ubuntu@YOUR_EC2_IP

# Check container
docker ps
docker logs fake-news-api

# Restart
docker restart fake-news-api

# Update code
cd Fake_News_Detector
git pull
docker stop fake-news-api
docker rm fake-news-api
docker build -t fake-news-detector .
docker run -d --name fake-news-api -p 80:5000 --restart unless-stopped fake-news-detector

# Monitor resources
htop  # Install: sudo apt install htop
docker stats
```

---

## 🎉 You're Done!

Your Fake News Detector API is now:
- ✅ Running on AWS EC2
- ✅ Accessible from anywhere
- ✅ Auto-restarts on failure
- ✅ Production-ready

**Access it at:** `http://YOUR_EC2_IP/api/check`

**Cost:** ~$36-40/month (or free for 12 months with t2.micro)

---

## 📞 Need Help?

- AWS Support: [aws.amazon.com/support](https://aws.amazon.com/support)
- Docker Docs: [docs.docker.com](https://docs.docker.com)
- Your logs: `docker logs fake-news-api`

**Happy Deploying! 🚀**
