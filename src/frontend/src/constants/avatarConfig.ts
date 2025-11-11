export interface AvatarOption {
  character: string;
  style: string;
  label: string;
  imagePath: string;
}

export interface VoiceOption {
  id: string;
  name: string;
  locale: string;
}

export interface LanguageOption {
  id: string;
  name: string;
  locale: string;
  voices: VoiceOption[];
}

// Avatar configurations with image paths
export const AVATAR_OPTIONS: AvatarOption[] = [
  { character: "harry", style: "business", label: "Harry - Business", imagePath: "/static/assets/avatar/harry/harry-business-thumbnail.png" },
  { character: "harry", style: "casual", label: "Harry - Casual", imagePath: "/static/assets/avatar/harry/harry-casual-thumbnail.png" },
  { character: "harry", style: "youthful", label: "Harry - Youthful", imagePath: "/static/assets/avatar/harry/harry-youthful-thumbnail.png" },
  { character: "jeff", style: "business", label: "Jeff - Business", imagePath: "/static/assets/avatar/jeff/jeff-business-thumbnail-bg.png" },
  { character: "jeff", style: "formal", label: "Jeff - Formal", imagePath: "/static/assets/avatar/jeff/jeff-formal-thumbnail-bg.png" },
  { character: "lisa", style: "casual-sitting", label: "Lisa - Casual Sitting", imagePath: "/static/assets/avatar/lisa/lisa-casual-sitting-thumbnail.png" },
  { character: "lisa", style: "graceful-sitting", label: "Lisa - Graceful Sitting", imagePath: "/static/assets/avatar/lisa/lisa-graceful-sitting-thumbnail.png" },
  { character: "lisa", style: "graceful-standing", label: "Lisa - Graceful Standing", imagePath: "/static/assets/avatar/lisa/lisa-graceful-standing-thumbnail.png" },
  { character: "lisa", style: "technical-sitting", label: "Lisa - Technical Sitting", imagePath: "/static/assets/avatar/lisa/lisa-technical-sitting-thumbnail.png" },
  { character: "lisa", style: "technical-standing", label: "Lisa - Technical Standing", imagePath: "/static/assets/avatar/lisa/lisa-technical-standing-thumbnail.png" },
  { character: "lori", style: "casual", label: "Lori - Casual", imagePath: "/static/assets/avatar/lori/lori-casual-thumbnail.png" },
  { character: "lori", style: "formal", label: "Lori - Formal", imagePath: "/static/assets/avatar/lori/lori-formal-thumbnail.png" },
  { character: "lori", style: "graceful", label: "Lori - Graceful", imagePath: "/static/assets/avatar/lori/lori-graceful-thumbnail.png" },
  { character: "max", style: "business", label: "Max - Business", imagePath: "/static/assets/avatar/max/max-business-thumbnail.png" },
  { character: "max", style: "casual", label: "Max - Casual", imagePath: "/static/assets/avatar/max/max-casual-thumbnail.png" },
  { character: "max", style: "formal", label: "Max - Formal", imagePath: "/static/assets/avatar/max/max-formal-thumbnail.png" },
  { character: "meg", style: "business", label: "Meg - Business", imagePath: "/static/assets/avatar/meg/meg-business-thumbnail.png" },
  { character: "meg", style: "casual", label: "Meg - Casual", imagePath: "/static/assets/avatar/meg/meg-casual-thumbnail.png" },
  { character: "meg", style: "formal", label: "Meg - Formal", imagePath: "/static/assets/avatar/meg/meg-formal-thumbnail.png" },
];

// Language and voice configurations
export const LANGUAGE_OPTIONS: LanguageOption[] = [
  {
    id: "en-GB",
    name: "English (United Kingdom)",
    locale: "en-GB",
    voices: [
      { id: "en-GB-AdaMultilingual", name: "Ada Multilingual", locale: "en-GB" },
      { id: "en-GB-OllieMultilingual", name: "Ollie Multilingual", locale: "en-GB" },
      { id: "en-GB-SoniaNeural", name: "Sonia", locale: "en-GB" },
      { id: "en-GB-RyanNeural", name: "Ryan", locale: "en-GB" },
      { id: "en-GB-LibbyNeural", name: "Libby", locale: "en-GB" },
      { id: "en-GB-AbbiNeural", name: "Abbi", locale: "en-GB" },
      { id: "en-GB-AlfieNeural", name: "Alfie", locale: "en-GB" },
      { id: "en-GB-BellaNeural", name: "Bella", locale: "en-GB" },
      { id: "en-GB-ElliotNeural", name: "Elliot", locale: "en-GB" },
      { id: "en-GB-EthanNeural", name: "Ethan", locale: "en-GB" },
      { id: "en-GB-HollieNeural", name: "Hollie", locale: "en-GB" },
      { id: "en-GB-MaisieNeural", name: "Maisie", locale: "en-GB" },
      { id: "en-GB-NoahNeural", name: "Noah", locale: "en-GB" },
      { id: "en-GB-OliverNeural", name: "Oliver", locale: "en-GB" },
      { id: "en-GB-OliviaNeural", name: "Olivia", locale: "en-GB" },
      { id: "en-GB-ThomasNeural", name: "Thomas", locale: "en-GB" },
    ],
  },
  {
    id: "en-US",
    name: "English (United States)",
    locale: "en-US",
    voices: [
      { id: "en-US-AvaMultilingual", name: "Ava Multilingual", locale: "en-US" },
      { id: "en-US-AndrewMultilingual", name: "Andrew Multilingual", locale: "en-US" },
      { id: "en-US-AmandaMultilingual", name: "Amanda Multilingual", locale: "en-US" },
      { id: "en-US-AdamMultilingual", name: "Adam Multilingual", locale: "en-US" },
      { id: "en-US-PhoebeMultilingual", name: "Phoebe Multilingual", locale: "en-US" },
      { id: "en-US-AlloyTurboMultilingual", name: "Alloy Turbo Multilingual", locale: "en-US" },
      { id: "en-US-NovaTurboMultilingual", name: "Nova Turbo Multilingual", locale: "en-US" },
      { id: "en-US-CoraMultilingual", name: "Cora Multilingual", locale: "en-US" },
      { id: "en-US-ChristopherMultilingual", name: "Christopher Multilingual", locale: "en-US" },
      { id: "en-US-BrandonMultilingual", name: "Brandon Multilingual", locale: "en-US" },
      { id: "en-US-DerekMultilingual", name: "Derek Multilingual", locale: "en-US" },
      { id: "en-US-DustinMultilingual", name: "Dustin Multilingual", locale: "en-US" },
      { id: "en-US-LewisMultilingual", name: "Lewis Multilingual", locale: "en-US" },
      { id: "en-US-LolaMultilingual", name: "Lola Multilingual", locale: "en-US" },
      { id: "en-US-NancyMultilingual", name: "Nancy Multilingual", locale: "en-US" },
      { id: "en-US-SerenaMultilingual", name: "Serena Multilingual", locale: "en-US" },
      { id: "en-US-SteffanMultilingual", name: "Steffan Multilingual", locale: "en-US" },
      { id: "en-US-EmmaMultilingual", name: "Emma Multilingual", locale: "en-US" },
      { id: "en-US-BrianMultilingual", name: "Brian Multilingual", locale: "en-US" },
      { id: "en-US-GuyNeural", name: "Guy", locale: "en-US" },
      { id: "en-US-AriaNeural", name: "Aria", locale: "en-US" },
      { id: "en-US-JaneNeural", name: "Jane", locale: "en-US" },
      { id: "en-US-JasonNeural", name: "Jason", locale: "en-US" },
      { id: "en-US-BrandonNeural", name: "Brandon", locale: "en-US" },
      { id: "en-US-ChristopherNeural", name: "Christopher", locale: "en-US" },
      { id: "en-US-CoraNeural", name: "Cora", locale: "en-US" },
      { id: "en-US-JennyMultilingual", name: "Jenny Multilingual", locale: "en-US" },
      { id: "en-US-RyanMultilingual", name: "Ryan Multilingual", locale: "en-US" },
      { id: "en-US-EchoTurboMultilingual", name: "Echo Turbo Multilingual", locale: "en-US" },
      { id: "en-US-FableTurboMultilingual", name: "Fable Turbo Multilingual", locale: "en-US" },
      { id: "en-US-OnyxTurboMultilingual", name: "Onyx Turbo Multilingual", locale: "en-US" },
      { id: "en-US-ShimmerTurboMultilingual", name: "Shimmer Turbo Multilingual", locale: "en-US" },
      { id: "en-US-AvaNeural", name: "Ava", locale: "en-US" },
      { id: "en-US-AndrewNeural", name: "Andrew", locale: "en-US" },
      { id: "en-US-EmmaNeural", name: "Emma", locale: "en-US" },
      { id: "en-US-BrianNeural", name: "Brian", locale: "en-US" },
      { id: "en-US-JennyNeural", name: "Jenny", locale: "en-US" },
      { id: "en-US-DavisNeural", name: "Davis", locale: "en-US" },
      { id: "en-US-KaiNeural", name: "Kai", locale: "en-US" },
      { id: "en-US-LunaNeural", name: "Luna", locale: "en-US" },
      { id: "en-US-SaraNeural", name: "Sara", locale: "en-US" },
      { id: "en-US-TonyNeural", name: "Tony", locale: "en-US" },
      { id: "en-US-NancyNeural", name: "Nancy", locale: "en-US" },
      { id: "en-US-AmberNeural", name: "Amber", locale: "en-US" },
      { id: "en-US-AnaNeural", name: "Ana", locale: "en-US" },
      { id: "en-US-AshleyNeural", name: "Ashley", locale: "en-US" },
      { id: "en-US-DavisMultilingual", name: "Davis Multilingual", locale: "en-US" },
      { id: "en-US-ElizabethNeural", name: "Elizabeth", locale: "en-US" },
      { id: "en-US-EricNeural", name: "Eric", locale: "en-US" },
      { id: "en-US-JacobNeural", name: "Jacob", locale: "en-US" },
      { id: "en-US-MichelleNeural", name: "Michelle", locale: "en-US" },
      { id: "en-US-MonicaNeural", name: "Monica", locale: "en-US" },
      { id: "en-US-RogerNeural", name: "Roger", locale: "en-US" },
      { id: "en-US-SamuelMultilingual", name: "Samuel Multilingual", locale: "en-US" },
      { id: "en-US-SteffanNeural", name: "Steffan", locale: "en-US" },
    ],
  },
];

// Default values
export const DEFAULT_AVATAR: AvatarOption = AVATAR_OPTIONS.find(
  (a) => a.character === "meg" && a.style === "business"
) || AVATAR_OPTIONS[0];

export const DEFAULT_LANGUAGE: LanguageOption = LANGUAGE_OPTIONS.find(
  (l) => l.id === "en-GB"
) || LANGUAGE_OPTIONS[0];

export const DEFAULT_VOICE: VoiceOption = DEFAULT_LANGUAGE.voices.find(
  (v) => v.id === "en-GB-LibbyNeural"
) || DEFAULT_LANGUAGE.voices[0];

