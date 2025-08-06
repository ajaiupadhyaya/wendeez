# Elite Fantasy Football Predictor - Deployment Guide

## 🚀 Complete Deployment Instructions

This guide covers deployment for development, staging, and production environments.

---

## Prerequisites

- Python 3.8+ installed
- Node.js 18+ installed  
- PostgreSQL 13+ (for production) or SQLite (development)
- Git
- Domain name (for production)
- Cloud hosting account (AWS, GCP, Azure, or DigitalOcean)

---

## 🔧 Local Development Setup

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd wendeez

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements-minimal.txt

# Initialize the system
python initialize.py
```

### 2. Start Backend API

```bash
# Terminal 1 - Start FastAPI server
source venv/bin/activate
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Start Frontend

```bash
# Terminal 2 - Start React development server
cd frontend
npm install
npm run dev
```

Access the application at: `http://localhost:5173`

---

## 🌐 Production Deployment

### Option 1: Deploy to AWS EC2

#### 1. Setup EC2 Instance

```bash
# Launch EC2 instance (Ubuntu 22.04 LTS recommended)
# - Instance type: t3.medium or larger
# - Storage: 20GB minimum
# - Security groups: Allow ports 22, 80, 443, 8000

# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv nginx postgresql postgresql-contrib nodejs npm git
```

#### 2. Setup PostgreSQL Database

```bash
# Configure PostgreSQL
sudo -u postgres psql

CREATE DATABASE fantasy_football_prod;
CREATE USER ff_user WITH PASSWORD 'your-secure-password';
GRANT ALL PRIVILEGES ON DATABASE fantasy_football_prod TO ff_user;
\q

# Update database config in config/production.yaml
```

#### 3. Deploy Backend

```bash
# Clone repository
git clone <your-repo-url> /home/ubuntu/wendeez
cd /home/ubuntu/wendeez

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service for API
sudo nano /etc/systemd/system/fantasy-api.service
```

Add this content:

```ini
[Unit]
Description=Fantasy Football API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/wendeez
Environment="PATH=/home/ubuntu/wendeez/venv/bin"
Environment="ENVIRONMENT=production"
ExecStart=/home/ubuntu/wendeez/venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable fantasy-api
sudo systemctl start fantasy-api
```

#### 4. Build and Deploy Frontend

```bash
cd /home/ubuntu/wendeez/frontend
npm install
npm run build

# Copy build to nginx directory
sudo cp -r dist/* /var/www/html/
```

#### 5. Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/fantasy-football
```

Add this configuration:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Frontend
    location / {
        root /var/www/html;
        try_files $uri $uri/ /index.html;
    }

    # API Proxy
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/fantasy-football /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### 6. Setup SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### Option 2: Deploy to Heroku

#### 1. Install Heroku CLI

```bash
# Install Heroku CLI
curl https://cli-assets.heroku.com/install.sh | sh
```

#### 2. Create Heroku App

```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:hobby-dev
```

#### 3. Create Procfile

```bash
echo "web: uvicorn src.api.app:app --host 0.0.0.0 --port $PORT" > Procfile
```

#### 4. Deploy

```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Run initialization
heroku run python initialize.py
```

---

### Option 3: Deploy with Docker

#### 1. Create Dockerfile for Backend

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Create Dockerfile for Frontend

```dockerfile
# frontend/Dockerfile
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

#### 3. Create docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: fantasy_football
      POSTGRES_USER: ff_user
      POSTGRES_PASSWORD: your-password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  backend:
    build: .
    environment:
      DATABASE_URL: postgresql://ff_user:your-password@postgres:5432/fantasy_football
      ENVIRONMENT: production
    ports:
      - "8000:8000"
    depends_on:
      - postgres

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
```

#### 4. Deploy with Docker

```bash
docker-compose up -d
```

---

### Option 4: Deploy to DigitalOcean App Platform

#### 1. Create App Specification

```yaml
# app.yaml
name: fantasy-football-predictor
region: nyc
services:
- name: api
  github:
    repo: your-github-username/your-repo
    branch: main
    deploy_on_push: true
  run_command: uvicorn src.api.app:app --host 0.0.0.0 --port 8080
  envs:
  - key: ENVIRONMENT
    value: production
  http_port: 8080
  instance_count: 1
  instance_size_slug: basic-xs

- name: frontend
  github:
    repo: your-github-username/your-repo
    branch: main
    deploy_on_push: true
  build_command: cd frontend && npm install && npm run build
  run_command: cd frontend && npm start
  http_port: 3000
  routes:
  - path: /

databases:
- name: fantasy-db
  engine: PG
  version: "15"
```

#### 2. Deploy

```bash
doctl apps create --spec app.yaml
```

---

## 📊 Monitoring & Maintenance

### Setup Monitoring

```bash
# Install monitoring tools
pip install prometheus-client grafana

# Add health check endpoint (already included in API)
# Monitor at: http://your-domain.com/api/health
```

### Database Backups

```bash
# Create backup script
nano backup.sh
```

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U ff_user fantasy_football_prod > backup_$DATE.sql
# Upload to S3 or other storage
aws s3 cp backup_$DATE.sql s3://your-bucket/backups/
```

```bash
# Setup cron job for daily backups
crontab -e
# Add: 0 2 * * * /home/ubuntu/backup.sh
```

### Update Data

```bash
# Create update script
nano update_data.sh
```

```bash
#!/bin/bash
cd /home/ubuntu/wendeez
source venv/bin/activate
python -m src.data.collectors.nfl_data_collector
```

```bash
# Setup weekly data updates
crontab -e
# Add: 0 6 * * 1 /home/ubuntu/update_data.sh
```

---

## 🔒 Security Best Practices

1. **Environment Variables**: Never commit sensitive data
```bash
# Create .env file
cp .env.example .env
# Edit with your values
```

2. **API Rate Limiting**: Already implemented in FastAPI

3. **HTTPS**: Always use SSL in production

4. **Database Security**: 
   - Use strong passwords
   - Restrict database access to application only
   - Regular backups

5. **Monitoring**: Set up alerts for:
   - High CPU/Memory usage
   - API errors
   - Database connection issues

---

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancer (AWS ELB, Nginx)
- Multiple API instances
- Redis for caching

### Database Optimization
```sql
-- Create indexes for performance
CREATE INDEX idx_game_stats_player ON game_stats(player_id);
CREATE INDEX idx_game_stats_season_week ON game_stats(season, week);
CREATE INDEX idx_players_position ON players(position);
```

### CDN for Frontend
- Use CloudFlare or AWS CloudFront
- Cache static assets
- Compress images and JS/CSS

---

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Error**
```bash
# Check PostgreSQL status
sudo systemctl status postgresql
# Check logs
sudo journalctl -u postgresql
```

2. **API Not Responding**
```bash
# Check service status
sudo systemctl status fantasy-api
# View logs
sudo journalctl -u fantasy-api -n 100
```

3. **Frontend Build Errors**
```bash
# Clear cache and rebuild
rm -rf node_modules package-lock.json
npm install
npm run build
```

4. **Model Prediction Errors**
```bash
# Retrain models
python -m src.models.ensemble_predictor
```

---

## 📞 Support & Maintenance

- Monitor application logs daily
- Update NFL data weekly (automated)
- Retrain models monthly
- Security updates monthly
- Full backups weekly

---

## 🎯 Performance Benchmarks

Expected performance metrics:
- API Response Time: < 200ms
- Prediction Generation: < 500ms
- Page Load Time: < 2s
- Uptime: 99.9%

---

## 🔄 CI/CD Pipeline (Optional)

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /home/ubuntu/wendeez
          git pull
          source venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart fantasy-api
          cd frontend
          npm install
          npm run build
          sudo cp -r dist/* /var/www/html/
```

---

## ✅ Post-Deployment Checklist

- [ ] API health check passing
- [ ] Frontend loading correctly
- [ ] Database connected
- [ ] SSL certificate active
- [ ] Monitoring setup
- [ ] Backup automation configured
- [ ] Data update scheduled
- [ ] Error logging enabled
- [ ] Rate limiting active
- [ ] Security headers configured

---

## 📝 Notes

- Always test in staging before production
- Keep backups before major updates
- Monitor resource usage and scale as needed
- Document any custom configurations

For questions or issues, check the logs first, then refer to this guide.

**Happy Deploying! 🚀**