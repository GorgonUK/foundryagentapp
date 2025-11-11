import React, { useCallback, useEffect, useState } from "react";
import { AgentPreview } from "./agents/AgentPreview";
import { ThemeProvider } from "./core/theme/ThemeProvider";
import {
  AvatarOption,
  LanguageOption,
  VoiceOption,
  DEFAULT_AVATAR,
  DEFAULT_LANGUAGE,
  DEFAULT_VOICE,
} from "../constants/avatarConfig";

const App: React.FC = () => {
  // State to store the agent details
  const [agentDetails, setAgentDetails] = useState({
    id: "loading",
    object: "agent",
    created_at: Date.now(),
    name: "Loading...",
    description: "Loading agent details...",
    model: "default",
    metadata: {
      logo: "robot",
    },
  });
  const [audioInputDeviceId, setAudioInputDeviceId] = useState<string | undefined>(undefined);
  const [selectedAvatar, setSelectedAvatar] = useState<AvatarOption>(DEFAULT_AVATAR);
  const [selectedLanguage, setSelectedLanguage] = useState<LanguageOption>(DEFAULT_LANGUAGE);
  const [selectedVoice, setSelectedVoice] = useState<VoiceOption>(DEFAULT_VOICE);

  const fetchAgentDetails = useCallback(async () => {
    try {
      const response = await fetch("/agent", {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
      });

      if (response.ok) {
        const data = await response.json();
        setAgentDetails(data);
      } else {
        console.error("Failed to fetch agent details");
        // Set fallback data if fetch fails
        setAgentDetails({
          id: "fallback",
          object: "agent",
          created_at: Date.now(),
          name: "AI Agent",
          description: "Could not load agent details",
          model: "default",
          metadata: {
            logo: "robot",
          },
        });
      }
    } catch (error) {
      console.error("Error fetching agent details:", error);
      // Set fallback data if fetch fails
      setAgentDetails({
        id: "error",
        object: "agent",
        created_at: Date.now(),
        name: "AI Agent",
        description: "Error loading agent details",
        model: "default",
        metadata: {
          logo: "robot",
        },
      });
    }
  }, []);

  // Fetch agent details when component mounts
  useEffect(() => {
    void fetchAgentDetails();
  }, [fetchAgentDetails]);

  const handleAgentChanged = useCallback(async (agentId: string) => {
    try {
      const res = await fetch("/agent/select", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ id: agentId }),
      });
      if (res.ok) {
        await fetchAgentDetails();
      } else {
        console.error("Failed to select agent");
      }
    } catch (e) {
      console.error("Error selecting agent:", e);
    }
  }, [fetchAgentDetails]);

  const handleAudioInputChanged = useCallback((deviceId?: string) => {
    setAudioInputDeviceId(deviceId || undefined);
  }, []);

  const handleAvatarChanged = useCallback((avatar: AvatarOption) => {
    setSelectedAvatar(avatar);
  }, []);

  const handleLanguageChanged = useCallback((language: LanguageOption) => {
    setSelectedLanguage(language);
    // Reset voice to first voice in the new language
    const firstVoice = language.voices[0];
    if (firstVoice) {
      setSelectedVoice(firstVoice);
    }
  }, []);

  const handleVoiceChanged = useCallback((voice: VoiceOption) => {
    setSelectedVoice(voice);
  }, []);

  return (
    <ThemeProvider>
      <div className="app-container">
        <AgentPreview
          resourceId="sample-resource-id"
          agentDetails={agentDetails}
          onAgentChanged={handleAgentChanged}
          audioInputDeviceId={audioInputDeviceId}
          onAudioInputChanged={handleAudioInputChanged}
          selectedAvatar={selectedAvatar}
          onAvatarChanged={handleAvatarChanged}
          selectedLanguage={selectedLanguage}
          onLanguageChanged={handleLanguageChanged}
          selectedVoice={selectedVoice}
          onVoiceChanged={handleVoiceChanged}
        />
      </div>
    </ThemeProvider>
  );
};

export default App;
