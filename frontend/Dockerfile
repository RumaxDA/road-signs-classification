# LTS Version 
FROM node:20-alpine

WORKDIR /app

# Copy packages
COPY package*.json ./ 

# Install packages
RUN npm install

# Copy other files
COPY . .

# Port 
EXPOSE 5173

# --host: Export port to world
CMD ["npm", "run", "dev", "--", "--host"]