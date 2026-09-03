import { useRef, useState } from "react";
import { Mic, Pause, Play, Volume2 } from "lucide-react";
import { Card, CardBody, CardHeader } from "../ui/Card";
import { Badge } from "../ui/Badge";
import { mediaUrl } from "@/lib/api";
import { cn } from "@/lib/cn";

const VOICE_REASONS: Record<string, string> = {
  tts_disabled: "Voice synthesis is disabled on the backend (TTS_ENABLED=false).",
  tts_generation_failed: "The local speech engine could not render this message.",
  empty_message: "No customer message was produced for this step.",
};

export function CustomerMessageCard({
  message,
  audioUrl,
  voiceReason,
  voiceEngine,
}: {
  message: string | null;
  audioUrl?: string | null;
  voiceReason?: string | null;
  voiceEngine?: string | null;
}) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);

  if (!message) {
    return (
      <Card>
        <CardHeader eyebrow="Customer communication" title="Customer message" />
        <CardBody>
          <p className="text-[13px] text-ink-muted">
            No customer message was generated for this recovery.
          </p>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        eyebrow="Customer communication"
        title="Customer message"
        action={
          <Badge tone="accent">Hinglish · AI-written</Badge>
        }
      />
      <CardBody className="space-y-4">
        <blockquote className="rounded-control border border-line/[.08] bg-surface-2 p-4 text-[14px] leading-relaxed text-ink">
          {message}
        </blockquote>

        {audioUrl ? (
          <div className="flex items-center gap-3 rounded-control border border-line/[.08] bg-bg/50 p-3">
            <button
              onClick={() => {
                const el = audioRef.current;
                if (!el) return;
                if (el.paused) {
                  void el.play();
                } else {
                  el.pause();
                }
              }}
              aria-label={playing ? "Pause voice artifact" : "Play voice artifact"}
              className={cn(
                "grid size-9 shrink-0 place-items-center rounded-full bg-accent text-white transition-transform hover:scale-105 active:scale-95",
              )}
            >
              {playing ? <Pause size={15} /> : <Play size={15} className="ml-0.5" />}
            </button>
            <div className="min-w-0 flex-1">
              <p className="flex items-center gap-1.5 text-[13px] font-medium text-ink">
                <Mic size={13} className="text-accent" />
                Voice artifact generated
              </p>
              <p className="text-2xs text-ink-faint">
                {voiceEngine ? `${voiceEngine} · ` : ""}audio file only — not a phone
                call, not delivered to the customer.
              </p>
            </div>
            <Volume2 size={15} className="shrink-0 text-ink-faint" />
            <audio
              ref={audioRef}
              src={mediaUrl(audioUrl)}
              onPlay={() => setPlaying(true)}
              onPause={() => setPlaying(false)}
              onEnded={() => setPlaying(false)}
              preload="none"
            />
          </div>
        ) : (
          <div className="rounded-control border border-line/[.08] bg-surface-2 p-3 text-2xs text-ink-muted">
            <span className="font-medium text-ink-faint">No voice artifact.</span>{" "}
            {voiceReason ? VOICE_REASONS[voiceReason] ?? voiceReason : "Not generated for this run."}
          </div>
        )}
      </CardBody>
    </Card>
  );
}
