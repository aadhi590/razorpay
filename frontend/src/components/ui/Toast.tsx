import {
  createContext,
  useCallback,
  useContext,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Check, Info, TriangleAlert, X } from "lucide-react";
import { cn } from "@/lib/cn";

type Kind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: Kind;
  message: string;
}
interface ToastCtx {
  push: (message: string, kind?: Kind) => void;
}
const Ctx = createContext<ToastCtx>({ push: () => {} });

const ICONS: Record<Kind, typeof Check> = {
  success: Check,
  error: TriangleAlert,
  info: Info,
};
const TONE: Record<Kind, string> = {
  success: "text-success",
  error: "text-danger",
  info: "text-accent",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const seq = useRef(0);

  const push = useCallback((message: string, kind: Kind = "info") => {
    const id = ++seq.current;
    setToasts((t) => [...t, { id, kind, message }]);
    window.setTimeout(() => {
      setToasts((t) => t.filter((x) => x.id !== id));
    }, 3600);
  }, []);

  return (
    <Ctx.Provider value={{ push }}>
      {children}
      <div className="pointer-events-none fixed bottom-5 right-5 z-[200] flex w-[min(360px,calc(100vw-2.5rem))] flex-col gap-2">
        <AnimatePresence>
          {toasts.map((t) => {
            const Icon = ICONS[t.kind];
            return (
              <motion.div
                key={t.id}
                initial={{ opacity: 0, y: 12, scale: 0.98 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, x: 24 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="panel pointer-events-auto flex items-start gap-2.5 px-3.5 py-3 shadow-pop"
              >
                <Icon size={16} className={cn("mt-0.5 shrink-0", TONE[t.kind])} />
                <p className="flex-1 text-sm leading-snug text-ink">{t.message}</p>
                <button
                  aria-label="Dismiss"
                  onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}
                  className="link-quiet -m-1 p-1"
                >
                  <X size={14} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </Ctx.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useToast = () => useContext(Ctx);
