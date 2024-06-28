import { useState } from "react";

function Hint({
  children,
  width,
}: {
  children: React.ReactNode;
  width?: string;
}) {
  // Font Awesome Free 6.5.2 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.

  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="Hint">
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="15px"
        height="15px"
        viewBox="0 0 15 15"
        onMouseEnter={() => setIsOpen(true)}
        onClick={() => setIsOpen(true)}
        onMouseLeave={() => setIsOpen(false)}
        onBlur={() => setIsOpen(false)}
      >
        <path
          fill="currentColor"
          fill-rule="evenodd"
          d="M0 7.5a7.5 7.5 0 1 1 15 0a7.5 7.5 0 0 1-15 0M5.5 6a2 2 0 0 1 2-2h.6c1.05 0 1.9.85 1.9 1.9V6a2 2 0 0 1-2 2v1H7V7h1a1 1 0 0 0 1-1v-.1a.9.9 0 0 0-.9-.9h-.6a1 1 0 0 0-1 1zM7 11v-1h1v1z"
          clip-rule="evenodd"
        />
      </svg>
      {isOpen && (
        <div className="Hint-text" style={{ width }}>
          {children}
        </div>
      )}
    </div>
  );
}

export default Hint;
