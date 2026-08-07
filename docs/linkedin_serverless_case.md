# Migrando para o Serverless: Resolvendo o Gargalo de Ingestão de Dados no RAG

Recentemente me deparei com um desafio clássico de escalabilidade no **Project Aether** (meu motor de busca semântica RAG). 

O problema: A API principal em FastAPI era super rápida para responder perguntas. Porém, a esteira de ingestão de novos documentos – que faz o *chunking*, limpeza de dados sensíveis (PII) com IA e geração de *embeddings* – rodava em uma fila local (RQ). Sempre que eu subia dezenas de PDFs pesados de uma vez, a CPU do servidor batia no teto e o processamento formava uma fila gigantesca.

A solução? Uma **Arquitetura Híbrida Orientada a Eventos (Event-Driven)**.

Decidi desacoplar o trabalho pesado e movê-lo para a AWS. Agora, o fluxo funciona assim:
1️⃣ O usuário faz upload do PDF pela aplicação. O arquivo é salvo diretamente no **Amazon S3**.
2️⃣ O simples ato do arquivo cair no S3 dispara automaticamente (*Trigger*) uma função **AWS Lambda**.
3️⃣ O Lambda acorda instantaneamente, baixa o arquivo, sanitiza os dados pessoais de forma paralela e salva os embeddings na Chroma Cloud. Logo em seguida, ele é desligado.

**O Resultado:**
Escalabilidade massiva. Se eu fizer upload de 1 PDF, 1 Lambda processa. Se eu subir 1.000 PDFs, a AWS sobe 1.000 Lambdas ao mesmo tempo e processa tudo simultaneamente em questão de segundos. O servidor principal do FastAPI? Continua consumindo 0% de CPU para essa tarefa, focando apenas em servir os usuários.

Ferramentas utilizadas: `Python`, `FastAPI`, `AWS Lambda`, `Amazon S3`, `Boto3` e `LlamaIndex`.

O código dessa refatoração Serverless já está no meu repositório do Github! Quem quiser dar uma olhada em como configurar a infraestrutura via código com o AWS SAM, o link está no primeiro comentário. 👇

#python #aws #serverless #backend #softwareengineering #rag #llm #cloudcomputing
