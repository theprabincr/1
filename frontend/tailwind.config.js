/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: '#09090B',
          paper: '#18181B',
          subtle: '#27272A',
        },
        text: {
          primary: '#FAFAFA',
          secondary: '#A1A1AA',
          muted: '#71717A',
        },
        brand: {
          primary: '#CCFF00',
          'primary-hover': '#B3E600',
          secondary: '#3B82F6',
        },
        semantic: {
          success: '#22C55E',
          danger: '#EF4444',
          warning: '#EAB308',
          info: '#3B82F6',
        },
        zinc: {
          700: '#3F3F46',
          800: '#27272A',
          900: '#18181B',
          950: '#09090B',
        },
        lime: {
          400: '#CCFF00',
        },
      },
      fontFamily: {
        sans: ['Manrope', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out forwards',
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
