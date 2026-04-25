/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // To mówi Tailwindowi: "Szukaj klas w tych plikach"
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}