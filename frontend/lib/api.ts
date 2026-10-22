const base = () => process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function newClientRequestId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `cid-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function apiFetch(
  path: string,
  opts: RequestInit & { token?: string | null } = {},
) {
  const headers = new Headers(opts.headers);
  headers.set("Content-Type", "application/json");
  headers.set("X-Client-Request-Id", newClientRequestId());
  if (opts.token) headers.set("Authorization", `Bearer ${opts.token}`);
  const { token: _t, ...rest } = opts;
  const r = await fetch(`${base()}${path}`, { ...rest, headers });
  return r;
}
