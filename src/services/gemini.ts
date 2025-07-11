import { GoogleGenAI } from "@google/genai";
import { env } from "../env.ts";

const gemini = new GoogleGenAI({
  apiKey: env.GEMINI_API_KEY,
});

const model = "gemini-2.5-flash";

export async function transcribeAudio(audioAsBase64: string, mimeType: string) {
  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: "Transcreva o áudio para o português do Brasil. Seja preciso e natural na transcrição. Mantenha a pontuação adequada e divida o texto em parágrafos quando for apropriado.",
      },
      {
        inlineData: {
          mimeType,
          data: audioAsBase64,
        },
      },
    ],
  });

  if (!response.text) {
    throw new Error("Failed to transcribe audio");
  }
  return response.text;
}

export async function generateEmbeddings(text: string) {
  const response = await gemini.models.embedContent({
    model: "text-embedding-004",
    contents: [
      {
        text,
      },
    ],
    config: {
      taskType: "RETRIEVAL_DOCUMENT",
    },
  });

  if (!response.embeddings?.[0].values) {
    throw new Error("Failed to generate embeddings");
  }

  return response.embeddings[0].values;
}

export async function generateAnswer(
  question: string,
  transcriptions: string[]
) {
  const context = transcriptions.join("\n\n");
  const prompt = `Responda a pergunta com base nas transcrições fornecidas.
  Seja preciso, claro e direto ao ponto. Em Português do Brasil.

  Pergunta: ${question}

  Contexto:
  ${context}

  Instruções:
  - Use as informações do contexto para responder.
  - Se a resposta não estiver no contexto, diga que não possui informações suficientes para responser.
  - Seja objetivo.
  - Mantenha um tom educativo e profissional.
  - Evite repetir a pergunta na resposta.
  - Cite trechos relevantes do contexto quando necessário.
  - Se for citar um trecho do contexto, utilize o termo "conteúdo da aula".
  `.trim();
  const response = await gemini.models.generateContent({
    model,
    contents: [
      {
        text: prompt,
      },
    ],
  });

  if (!response.text) {
    throw new Error("Failed to generate answer");
  }

  return response.text;
}
