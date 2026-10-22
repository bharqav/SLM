import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex max-w-lg flex-col gap-6 p-8">
      <h1 className="text-2xl font-semibold text-emerald-400">Emotional SLM</h1>
      <p className="text-slate-400">Sign in to start a supportive conversation.</p>
      <Link
        href="/login"
        className="rounded-lg bg-emerald-600 px-4 py-2 text-center font-medium text-white hover:bg-emerald-500"
      >
        Login / Sign up
      </Link>
      <Link href="/chat" className="text-center text-sm text-slate-500 underline">
        Open chat (requires auth)
      </Link>
    </main>
  );
}
