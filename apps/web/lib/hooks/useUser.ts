"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { api, ApiError, type User } from "../api";

/** Loads the signed-in user; redirects to /login when required and not signed in. */
export function useUser(required = true) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  useEffect(() => {
    let alive = true;
    api<User>("/auth/me")
      .then((u) => alive && setUser(u))
      .catch((e) => {
        if (required && e instanceof ApiError && e.status === 401) router.replace("/login");
      })
      .finally(() => alive && setLoading(false));
    return () => { alive = false; };
  }, [required, router]);
  return { user, loading };
}
