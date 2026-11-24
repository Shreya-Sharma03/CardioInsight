/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#FBF9F5",
        surface: "#FFFFFF",
        sand: "#F4F0EA",
        sandBorder: "#E5E0D8",
        charcoal: {
          DEFAULT: "#1A1C1E",
          muted: "#5A6065",
          subtle: "#7E858B",
        },
        sage: {
          DEFAULT: "#4A7C59",
          dark: "#3D7055",
          light: "#EDF4F0",
          border: "#CBE0D4",
        },
        terracotta: {
          DEFAULT: "#C05638",
          dark: "#A84428",
          light: "#FDF2EE",
          border: "#F6D3C8",
        },
      },
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"Segoe UI"',
          "Roboto",
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
        mono: ['"SF Mono"', "Menlo", "Monaco", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
}
