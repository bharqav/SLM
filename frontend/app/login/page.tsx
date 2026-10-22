"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { apiFetch } from "@/lib/api";
import { useAuthStore } from "@/lib/store";

export default function LoginPage() {
  const router = useRouter();
  const setTokens = useAuthStore((s) => s.setTokens);
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [err, setErr] = useState<string | null>(null);
  const [acceptTerms, setAcceptTerms] = useState(false);
  const [confirmAge, setConfirmAge] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(null);
    if (mode === "signup" && (!acceptTerms || !confirmAge)) {
      setErr("Please confirm age 18+ and accept the terms to continue.");
      return;
    }
    const path = mode === "login" ? "/api/v1/auth/login" : "/api/v1/auth/signup";
    const body =
      mode === "login"
        ? { email, password }
        : { email, username, password, accept_terms: acceptTerms, confirm_age_18: confirmAge };
    const r = await apiFetch(path, { method: "POST", body: JSON.stringify(body) });
    if (!r.ok) {
      setErr(await r.text());
      return;
    }
    const data = (await r.json()) as { access_token: string; refresh_token: string };
    setTokens(data.access_token, data.refresh_token);
    router.push("/chat");
  }

  return (
    <main className="mx-auto max-w-md p-8">
      <h1 className="mb-6 text-xl font-semibold">{mode === "login" ? "Login" : "Sign up"}</h1>
      <form onSubmit={submit} className="flex flex-col gap-3">
        <input
          className="rounded border border-slate-700 bg-slate-900 px-3 py-2"
          placeholder="Email"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        {mode === "signup" && (
          <>
            <input
              className="rounded border border-slate-700 bg-slate-900 px-3 py-2"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
            />
            <label className="flex cursor-pointer items-start gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                className="mt-1"
                checked={confirmAge}
                onChange={(e) => setConfirmAge(e.target.checked)}
              />
              <span>I confirm I am 18 or older. This app is not medical care or emergency services.</span>
            </label>
            <label className="flex cursor-pointer items-start gap-2 text-sm text-slate-300">
              <input
                type="checkbox"
                className="mt-1"
                checked={acceptTerms}
                onChange={(e) => setAcceptTerms(e.target.checked)}
              />
              <span>I accept the Terms of Service and Privacy Policy.</span>
            </label>
          </>
        )}
        <input
          className="rounded border border-slate-700 bg-slate-900 px-3 py-2"
          placeholder="Password"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {err && <p className="text-sm text-red-400">{err}</p>}
        <button type="submit" className="rounded bg-emerald-600 py-2 font-medium text-white">
          {mode === "login" ? "Login" : "Create account"}
        </button>
      </form>
      <button
        type="button"
        className="mt-4 text-sm text-slate-400 underline"
        onClick={() => setMode(mode === "login" ? "signup" : "login")}
      >
        {mode === "login" ? "Need an account? Sign up" : "Have an account? Login"}
      </button>
    </main>
  );
}
