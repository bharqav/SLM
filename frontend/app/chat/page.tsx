"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";
import { useAuthStore } from "@/lib/store";

type Conv = { id: string; title: string };

const MAX_MESSAGE_LEN = 32000;

type ChatMsg = {
  role: string;
  content: string;
  assistantMessageId?: string;
  transparencyNote?: string;
};

export default function ChatPage() {
  const router = useRouter();
  const { accessToken, logout } = useAuthStore();
  const [conversations, setConversations] = useState<Conv[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [streaming, setStreaming] = useState("");
  const [feedbackDone, setFeedbackDone] = useState<Record<string, true>>({});
  const [modelIssue, setModelIssue] = useState<string | null>(null);

  useEffect(() => {
    if (!accessToken) {
      router.replace("/login");
      return;
    }
    void (async () => {
      const r = await apiFetch("/api/v1/conversations", { token: accessToken });
      if (r.ok) {
        const raw = (await r.json()) as { items?: Conv[] } | Conv[];
        const list = Array.isArray(raw) ? raw : raw.items ?? [];
        setConversations(list);
      }
    })();
  }, [accessToken, router]);

  useEffect(() => {
    if (!activeId || typeof window === "undefined") return;
    const saved = localStorage.getItem(`slm_draft_${activeId}`);
    if (saved) setInput(saved);
    else setInput("");
  }, [activeId]);

  useEffect(() => {
    if (!activeId || typeof window === "undefined") return;
    const id = window.setTimeout(() => {
      localStorage.setItem(`slm_draft_${activeId}`, input);
    }, 400);
    return () => window.clearTimeout(id);
  }, [input, activeId]);

  async function newChat() {
    if (!accessToken) return;
    const r = await apiFetch("/api/v1/conversations", {
      method: "POST",
      body: JSON.stringify({ title: "Chat" }),
      token: accessToken,
    });
    if (r.ok) {
      const c = (await r.json()) as Conv;
      setConversations((x) => [c, ...x]);
      setActiveId(c.id);
      setMessages([]);
    }
  }

  function parseErrorMessage(text: string): string {
    try {
      const j = JSON.parse(text) as { error?: { message?: string } };
      return j.error?.message ?? text;
    } catch {
      return text;
    }
  }

  async function send() {
    if (!accessToken || !activeId || !input.trim()) return;
    const userText = input.trim();
    if (userText.length > MAX_MESSAGE_LEN) {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Message too long (max ${MAX_MESSAGE_LEN} characters).`,
        },
      ]);
      return;
    }
    setModelIssue(null);
    setInput("");
    localStorage.removeItem(`slm_draft_${activeId}`);
    setMessages((m) => [...m, { role: "user", content: userText }]);
    setStreaming("…");
    let r: Response | null = null;
    for (let attempt = 0; attempt < 3; attempt++) {
      r = await apiFetch("/api/v1/chat", {
        method: "POST",
        body: JSON.stringify({ conversation_id: activeId, content: userText }),
        token: accessToken,
      });
      if (r.ok) break;
      if (r.status >= 500 && attempt < 2) {
        await new Promise((res) => setTimeout(res, 400 * (attempt + 1)));
        continue;
      }
      break;
    }
    setStreaming("");
    if (!r?.ok) {
      const raw = r ? await r.text() : "Network error";
      const msg = parseErrorMessage(raw);
      const unavailable = r && (r.status === 502 || r.status === 503);
      setModelIssue(
        unavailable
          ? "The assistant is temporarily unavailable. You can retry in a moment."
          : null,
      );
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: unavailable ? `${msg} Tap Send again to retry.` : msg,
        },
      ]);
      return;
    }
    const data = (await r.json()) as {
      assistant_message: { id: string; content: string };
      transparency?: { constrained?: boolean; notes?: string[] };
    };
    const notes = data.transparency?.notes?.length
      ? data.transparency.notes.join(" ")
      : data.transparency?.constrained
        ? "This reply follows safety guidelines."
        : undefined;
    setMessages((m) => [
      ...m,
      {
        role: "assistant",
        content: data.assistant_message.content,
        assistantMessageId: data.assistant_message.id,
        transparencyNote: notes,
      },
    ]);
  }

  async function submitFeedback(messageId: string, rating: number) {
    if (!accessToken) return;
    const r = await apiFetch("/api/v1/feedback", {
      method: "POST",
      body: JSON.stringify({ message_id: messageId, rating }),
      token: accessToken,
    });
    if (!r.ok) return;
    setFeedbackDone((prev) => ({ ...prev, [messageId]: true }));
  }

  if (!accessToken) return null;

  return (
    <div className="flex min-h-screen">
      <aside className="w-56 border-r border-slate-800 p-4">
        <button
          onClick={newChat}
          className="mb-4 w-full rounded bg-emerald-700 py-2 text-sm font-medium text-white"
        >
          New chat
        </button>
        <ul className="space-y-1 text-sm">
          {conversations.map((c) => (
            <li key={c.id}>
              <button
                className={`w-full truncate rounded px-2 py-1 text-left ${
                  activeId === c.id ? "bg-slate-800" : "hover:bg-slate-900"
                }`}
                onClick={() => {
                  setActiveId(c.id);
                  setMessages([]);
                }}
              >
                {c.title}
              </button>
            </li>
          ))}
        </ul>
        <button onClick={() => logout()} className="mt-8 text-xs text-slate-500 underline">
          Log out
        </button>
      </aside>
      <main className="flex flex-1 flex-col p-4">
        {!activeId && <p className="text-slate-500">Select or create a conversation.</p>}
        {activeId && (
          <>
            <div
              className="sr-only"
              aria-live="polite"
              aria-atomic="true"
            >
              {streaming ? "Assistant is responding." : modelIssue ?? ""}
            </div>
            {modelIssue && (
              <div className="mb-2 rounded border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-sm text-amber-100">
                {modelIssue}
              </div>
            )}
            <div className="flex-1 space-y-3 overflow-y-auto">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`rounded-lg px-3 py-2 ${
                    m.role === "user" ? "ml-8 bg-slate-800" : "mr-8 bg-slate-900"
                  }`}
                >
                  <div className="whitespace-pre-wrap">{m.content}</div>
                  {m.role === "assistant" && m.transparencyNote && (
                    <p className="mt-2 border-t border-slate-700 pt-2 text-xs text-slate-500">{m.transparencyNote}</p>
                  )}
                  {m.role === "assistant" &&
                    m.assistantMessageId &&
                    !feedbackDone[m.assistantMessageId] && (
                    <div className="mt-2 flex flex-wrap items-center gap-1 border-t border-slate-700 pt-2 text-xs text-slate-400">
                      <span>Rate:</span>
                      {[1, 2, 3, 4, 5].map((star) => (
                        <button
                          key={star}
                          type="button"
                          className="rounded px-1.5 py-0.5 hover:bg-slate-800"
                          onClick={() => void submitFeedback(m.assistantMessageId!, star)}
                        >
                          {star}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
              {streaming && <div className="text-slate-500">{streaming}</div>}
            </div>
            <div className="mt-4 flex gap-2">
              <input
                className="flex-1 rounded border border-slate-700 bg-slate-900 px-3 py-2"
                value={input}
                maxLength={MAX_MESSAGE_LEN}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void send()}
                placeholder="Type a message…"
              />
              <button
                onClick={() => void send()}
                className="rounded bg-emerald-600 px-4 py-2 font-medium text-white"
              >
                Send
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
