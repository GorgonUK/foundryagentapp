import type { JSX } from "react";
import { useEffect, useState } from "react";
import {
  Button,
  Drawer,
  DrawerBody,
  DrawerHeader,
  DrawerHeaderTitle,
  Dropdown,
  Label,
  Option,
} from "@fluentui/react-components";
import { Dismiss24Regular } from "@fluentui/react-icons";

import styles from "./SettingsPanel.module.css";
import { ThemePicker } from "./theme/ThemePicker";

export interface ISettingsPanelProps {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
  currentAgentId?: string;
  onAgentChanged?: (agentId: string) => Promise<void> | void;
  audioInputDeviceId?: string;
  onAudioInputChange?: (deviceId?: string) => Promise<void> | void;
}

export function SettingsPanel({
  isOpen = false,
  onOpenChange,
  currentAgentId,
  onAgentChanged,
  audioInputDeviceId,
  onAudioInputChange,
}: ISettingsPanelProps): JSX.Element {
  const [agents, setAgents] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(currentAgentId);
  const [loadingAgents, setLoadingAgents] = useState(false);
  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedAudioId, setSelectedAudioId] = useState<string | undefined>(
    audioInputDeviceId
  );
  const [loadingAudioDevices, setLoadingAudioDevices] = useState(false);
  const [audioDeviceError, setAudioDeviceError] = useState<string | null>(null);

  useEffect(() => {
    setSelectedId(currentAgentId);
  }, [currentAgentId]);

  useEffect(() => {
    setSelectedAudioId(audioInputDeviceId);
  }, [audioInputDeviceId]);

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const loadAgents = async () => {
      setLoadingAgents(true);
      try {
        const res = await fetch("/agents", {
          method: "GET",
          headers: { "Content-Type": "application/json" },
          credentials: "include",
        });
        if (res.ok) {
          const data = await res.json();
          const list = (data?.agents ?? []) as Array<{ id: string; name: string }>;
          if (!cancelled) {
            setAgents(list);
            // Ensure a selection
            if (!selectedId && list.length > 0) {
              setSelectedId(list[0].id);
            }
          }
        }
      } catch {
        // ignore
      } finally {
        if (!cancelled) setLoadingAgents(false);
      }
    };
    void loadAgents();
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    let cancelled = false;

    const loadAudioDevices = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        setAudioDeviceError("Browser does not support audio device selection.");
        setAudioDevices([]);
        return;
      }

      setLoadingAudioDevices(true);
      setAudioDeviceError(null);

      try {
        let devices = await navigator.mediaDevices.enumerateDevices();
        let audioInputs = devices.filter((device) => device.kind === "audioinput");

        if (!cancelled && audioInputs.length > 0 && audioInputs.every((d) => !d.label)) {
          try {
            const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            devices = await navigator.mediaDevices.enumerateDevices();
            audioInputs = devices.filter((device) => device.kind === "audioinput");
            tempStream.getTracks().forEach((track) => track.stop());
          } catch (err) {
            console.warn("Unable to retrieve microphone labels without permission", err);
          }
        }

        if (cancelled) {
          return;
        }

        setAudioDevices(audioInputs);

        if (audioInputs.length === 0) {
          setAudioDeviceError("No audio input devices detected.");
        }

        if (
          audioInputDeviceId &&
          audioInputs.every((device) => device.deviceId !== audioInputDeviceId)
        ) {
          setSelectedAudioId(undefined);
          if (onAudioInputChange) {
            void onAudioInputChange(undefined);
          }
        }
      } catch (err) {
        if (!cancelled) {
          console.error("Failed to enumerate audio devices", err);
          setAudioDeviceError("Unable to load audio input devices.");
          setAudioDevices([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingAudioDevices(false);
        }
      }
    };

    void loadAudioDevices();

    return () => {
      cancelled = true;
    };
  }, [isOpen, audioInputDeviceId, onAudioInputChange]);

  const changeAgent = async (agentId: string) => {
    setSelectedId(agentId);
    if (onAgentChanged) {
      await onAgentChanged(agentId);
    }
  };

  const changeAudioDevice = async (deviceId: string) => {
    const normalized = deviceId === "default" ? undefined : deviceId;
    setSelectedAudioId(normalized);
    if (onAudioInputChange) {
      await onAudioInputChange(normalized);
    }
  };

  const selectedAudioDevice = selectedAudioId
    ? audioDevices.find((device) => device.deviceId === selectedAudioId)
    : undefined;
  const audioDropdownValue =
    selectedAudioDevice?.label?.trim() ||
    (loadingAudioDevices ? "Loading microphones..." : "System default");

  return (
    <Drawer
      className={styles.panel}
      onOpenChange={(_, { open }) => {
        onOpenChange(open);
      }}
      open={isOpen}
      position="end"
    >
      <DrawerHeader>
        <DrawerHeaderTitle
          action={
            <div>
              <Button
                appearance="subtle"
                icon={<Dismiss24Regular />}
                onClick={() => {
                  onOpenChange(false);
                }}
              />
            </div>
          }
        >
          Settings
        </DrawerHeaderTitle>
      </DrawerHeader>{" "}
      <DrawerBody className={styles.content}>
        <div className={styles.settingSection}>
          <ThemePicker />
        </div>
        <div className={styles.settingSection}>
          <Label htmlFor="activeAgentDropdown">Active agent</Label>
          <Dropdown
            id="activeAgentDropdown"
            value={
              agents.find((a) => a.id === selectedId)?.name ??
              (loadingAgents ? "Loading agents..." : "Select an agent")
            }
            selectedOptions={selectedId ? [selectedId] : []}
            onOptionSelect={(_, data) => {
              const newId = data.optionValue as string;
              void changeAgent(newId);
            }}
          >
            {agents.map((agent) => (
              <Option key={agent.id} value={agent.id}>
                {agent.name || agent.id}
              </Option>
            ))}
          </Dropdown>
        </div>
        <div className={styles.settingSection}>
          <Label htmlFor="audioInputDropdown">Audio input device</Label>
          <Dropdown
            id="audioInputDropdown"
            value={audioDropdownValue}
            selectedOptions={[selectedAudioId ?? "default"]}
            onOptionSelect={(_, data) => {
              const newId = data.optionValue as string;
              void changeAudioDevice(newId);
            }}
          >
            <Option key="default" value="default">
              System default
            </Option>
            {audioDevices.map((device, index) => (
              <Option key={`${device.deviceId}-${index}`} value={device.deviceId}>
                {device.label || `Microphone ${index + 1}`}
              </Option>
            ))}
          </Dropdown>
          {audioDeviceError && (
            <span className={styles.settingHint}>{audioDeviceError}</span>
          )}
        </div>

      </DrawerBody>
    </Drawer>
  );
}
