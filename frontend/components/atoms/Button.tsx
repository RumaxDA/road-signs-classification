interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  loading?: boolean;
}

export const Button = ({ children, loading, ...props }: Props) => (
  <button
    {...props}
    className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-8 py-3 rounded-full font-bold transition w-full max-w-md shadow-lg"
  >
    {loading ? "Przetwarzanie..." : children}
  </button>
);
