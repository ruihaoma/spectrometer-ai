export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "Microsoft YaHei", "PingFang SC", "Arial", "sans-serif"],
      },
      colors: {
        lab: {
          ink: "#172033",
          muted: "#667085",
          line: "#d8dee9",
          panel: "#ffffff",
          field: "#f7f9fc",
          blue: "#1f4f8b",
          blueSoft: "#e8f0fb",
          danger: "#b42318",
          success: "#067647",
        },
      },
    },
  },
  plugins: [],
};
