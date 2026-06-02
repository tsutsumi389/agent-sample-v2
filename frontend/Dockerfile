FROM node:22-slim

WORKDIR /app

COPY package.json ./
RUN npm install

COPY . .

EXPOSE 5173

# --host で 0.0.0.0 にバインドしコンテナ外からアクセス可能にする
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]
