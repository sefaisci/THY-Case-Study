interface BrandMarkProps {
  className?: string;
  size?: "small" | "medium" | "large";
}

export function BrandMark({ className = "", size = "medium" }: BrandMarkProps) {
  return (
    <span className={`brand-mark brand-mark--${size} ${className}`} aria-hidden="true">
      <img src="/thy-logo.png" alt="" />
    </span>
  );
}

export function BrandLockup() {
  return (
    <div className="brand-lockup" aria-label="Turkish Airlines Document Intelligence">
      <BrandMark size="large" />
      <div className="brand-lockup__copy">
        <strong>Turkish Airlines</strong>
        <span>Document Intelligence</span>
      </div>
    </div>
  );
}
