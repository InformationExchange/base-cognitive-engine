/**
 * Audit Mode Toggle Component
 * Matches Onyx UI design system
 * 
 * Proprietary IP - 100% owned by Invitas Inc.
 */

"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface AuditModeToggleProps {
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  className?: string;
}

export default function AuditModeToggle({
  enabled,
  onToggle,
  className,
}: AuditModeToggleProps) {
  return (
    <button
      onClick={() => onToggle(!enabled)}
      className={cn(
        "flex items-center gap-2 px-3 py-1.5 rounded-08",
        "text-sm font-medium transition-colors",
        "border border-border",
        enabled
          ? "bg-accent-background text-accent border-accent"
          : "bg-background-neutral-01 text-text-600 hover:bg-background-neutral-02",
        className
      )}
      aria-label={enabled ? "Disable audit mode" : "Enable audit mode"}
    >
      <svg
        className={cn("w-4 h-4", enabled ? "text-accent" : "text-text-600")}
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        {enabled ? (
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        ) : (
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
          />
        )}
      </svg>
      <span>{enabled ? "Audit Mode" : "Enable Audit"}</span>
    </button>
  );
}










