export function formatAgentName(name?: string | null): string {
  if (!name) {
    return "";
  }

  const withSpaces = name
    .replace(/[_-]+/g, " ")
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/(\d)([A-Za-z])/g, "$1 $2")
    .replace(/([A-Za-z])(\d)/g, "$1 $2");

  return withSpaces.replace(/\\s+/g, " ").trim();
}

