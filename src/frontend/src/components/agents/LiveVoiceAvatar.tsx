import { useEffect, useRef } from "react";
import styles from "./LiveVoiceAvatar.module.css";

interface LiveVoiceAvatarProps {
  videoStream: MediaStream | null;
  isConnected: boolean;
}

export function LiveVoiceAvatar({ 
  videoStream,
  isConnected
}: LiveVoiceAvatarProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && videoStream) {
      videoRef.current.srcObject = videoStream;
      // Ensure audio is not muted
      videoRef.current.muted = false;
      videoRef.current.volume = 1.0;
      videoRef.current.play().catch(e => {
        console.error("Error playing video:", e);
      });
    }
  }, [videoStream]);

  return (
    <div className={styles.avatarContainer}>
      <div className={styles.avatarWrapper}>
        {videoStream ? (
          <video
            ref={videoRef}
            className={styles.avatarVideo}
            autoPlay
            playsInline
            muted={false}
          />
        ) : (
          <div className={styles.avatarPlaceholder}>
            {isConnected ? (
              <div className={styles.loadingText}>Connecting avatar...</div>
            ) : (
              <div className={styles.loadingText}>Waiting for connection...</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
