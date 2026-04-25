interface Props {
  children: React.ReactNode;
  variant?: "primary" | "secondary" | "error" | "success";
  className?: string;
}

export const Text = ({
  children,
  variant = "primary",
  className = "",
}: Props) => {
  const colors = {
    primary: "text-gray-200",
    secondary: "text-gray-400",
    error: "text-red-400",
    success: "text-emerald-400",
  };

  return (
    <p className={`${colors[variant]} text-sm font-medium ${className}`}>
      {children}
    </p>
  );
};
