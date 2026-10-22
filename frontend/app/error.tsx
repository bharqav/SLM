"use client";

import { useEffect } from "react";

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <main className="flex min-h-screen items-center justify-center bg-slate-950 px-6 text-slate-100">
      <div className="w-full max-w-xl rounded-3xl border border-white/10 bg-white/5 p-8 shadow-2xl shadow-emerald-950/30">
        <p className="text-sm uppercase tracking-[0.35em] text-emerald-300">Application error</p>
        <h1 className="mt-3 text-3xl font-semibold">Something went wrong</h1>
        <p className="mt-4 text-sm leading-6 text-slate-300">
          The app hit an unexpected error. You can retry the action or reload the page.
        </p>
        <button
          type="button"
          onClick={reset}
          className="mt-6 rounded-full bg-emerald-500 px-5 py-2.5 text-sm font-medium text-slate-950 transition hover:bg-emerald-400"
        >
          Try again
        </button>
      </div>
    </main>
  );
}