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
}

export function SettingsPanel({
  isOpen = false,
  onOpenChange,
  currentAgentId,
  onAgentChanged,
}: ISettingsPanelProps): JSX.Element {
  const [agents, setAgents] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedId, setSelectedId] = useState<string | undefined>(currentAgentId);
  const [loadingAgents, setLoadingAgents] = useState(false);

  useEffect(() => {
    setSelectedId(currentAgentId);
  }, [currentAgentId]);

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

  const changeAgent = async (agentId: string) => {
    setSelectedId(agentId);
    if (onAgentChanged) {
      await onAgentChanged(agentId);
    }
  };

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
          <ThemePicker />
        </div>
      </DrawerBody>
    </Drawer>
  );
}
