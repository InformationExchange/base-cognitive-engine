/**
 * Cognitive Enhancement Comparison Dashboard
 * Matches Onyx UI design system
 * 
 * Proprietary IP - 100% owned by Invitas Inc.
 */

"use client";

import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import MinimalMarkdown from "@/components/chat/MinimalMarkdown";

interface GovernanceMetrics {
  confidence: number;
  rag_quality: number;
  fact_check_coverage: number;
  reasoning_valid: boolean;
  temporal_state?: string;
  behavioral_state?: string;
}

interface ComparisonResult {
  scenario_id: string;
  scenario_name: string;
  query: string;
  baseline_result: {
    response: string;
    confidence: number;
    reasoning_path: string;
    processing_time: number;
    governance_metrics: GovernanceMetrics;
  };
  enhanced_result: {
    response: string;
    confidence: number;
    reasoning_path: string;
    processing_time: number;
    governance_metrics: GovernanceMetrics;
  };
  improvement_metrics: {
    confidence_improvement: number;
    rag_quality_improvement: number;
    processing_time_change: number;
    reasoning_path_upgrade: number;
  };
  winner: "baseline" | "enhanced" | "tie";
  audit_record_id?: string;
}

interface CognitiveComparisonDashboardProps {
  comparison: ComparisonResult | null;
  isOpen: boolean;
  onClose: () => void;
  position?: "bottom" | "side";
}

export default function CognitiveComparisonDashboard({
  comparison,
  isOpen,
  onClose,
  position = "bottom",
}: CognitiveComparisonDashboardProps) {
  const [activeTab, setActiveTab] = useState<"comparison" | "metrics" | "details">("comparison");

  if (!comparison || !isOpen) return null;

  const { baseline_result, enhanced_result, improvement_metrics, winner } = comparison;

  const formatConfidence = (conf: number) => (conf * 100).toFixed(1);
  const formatTime = (time: number) => time.toFixed(2);
  const formatImprovement = (val: number) => {
    const sign = val >= 0 ? "+" : "";
    return `${sign}${(val * 100).toFixed(1)}%`;
  };

  const getWinnerColor = () => {
    if (winner === "enhanced") return "text-green-600 dark:text-green-400";
    if (winner === "baseline") return "text-orange-600 dark:text-orange-400";
    return "text-gray-600 dark:text-gray-400";
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 0.8) return "text-green-600 dark:text-green-400";
    if (conf >= 0.6) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  // Panel content
  const panelContent = (
    <div
      className={cn(
        "bg-background-neutral-00 rounded-16 shadow-01 border border-border",
        "overflow-hidden",
        position === "bottom" ? "w-full" : "w-full sm:w-[600px]"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-background-neutral-02">
        <div className="flex items-center gap-3">
          <h3 className="text-text-800 font-semibold text-base">
            Cognitive Enhancement Comparison
          </h3>
          <span className={cn("text-xs font-medium px-2 py-1 rounded-08", getWinnerColor(), "bg-background-neutral-01")}>
            {winner.toUpperCase()} WINS
          </span>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-08 hover:bg-background-neutral-03 transition-colors"
          aria-label="Close comparison"
        >
          <svg
            className="w-4 h-4 text-text-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border bg-background-neutral-01">
        <button
          onClick={() => setActiveTab("comparison")}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors",
            activeTab === "comparison"
              ? "text-text-800 border-b-2 border-accent"
              : "text-text-600 hover:text-text-800"
          )}
        >
          Comparison
        </button>
        <button
          onClick={() => setActiveTab("metrics")}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors",
            activeTab === "metrics"
              ? "text-text-800 border-b-2 border-accent"
              : "text-text-600 hover:text-text-800"
          )}
        >
          Metrics
        </button>
        <button
          onClick={() => setActiveTab("details")}
          className={cn(
            "px-4 py-2 text-sm font-medium transition-colors",
            activeTab === "details"
              ? "text-text-800 border-b-2 border-accent"
              : "text-text-600 hover:text-text-800"
          )}
        >
          Details
        </button>
      </div>

      {/* Content */}
      <div className="overflow-y-auto max-h-[600px]">
        {activeTab === "comparison" && (
          <div className="p-4 space-y-4">
            {/* Side-by-side comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Baseline */}
              <div className="border border-border rounded-12 p-4 bg-background-neutral-01">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-text-800">Baseline</h4>
                  <span className="text-xs text-text-600">No Enhancements</span>
                </div>
                <div className="space-y-2 mb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Confidence</span>
                    <span className={cn("text-sm font-medium", getConfidenceColor(baseline_result.confidence))}>
                      {formatConfidence(baseline_result.confidence)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Reasoning</span>
                    <span className="text-xs font-medium text-text-700 capitalize">
                      {baseline_result.reasoning_path}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Time</span>
                    <span className="text-xs text-text-700">
                      {formatTime(baseline_result.processing_time)}s
                    </span>
                  </div>
                </div>
                <div className="border-t border-border pt-3">
                  <div className="text-sm text-text-700 prose prose-sm max-w-none">
                    <MinimalMarkdown content={baseline_result.response.substring(0, 300) + "..."} />
                  </div>
                </div>
              </div>

              {/* Enhanced */}
              <div className="border border-accent rounded-12 p-4 bg-accent-background">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-semibold text-text-800">Enhanced</h4>
                  <span className="text-xs text-accent font-medium">With Cognitive Enhancements</span>
                </div>
                <div className="space-y-2 mb-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Confidence</span>
                    <span className={cn("text-sm font-medium", getConfidenceColor(enhanced_result.confidence))}>
                      {formatConfidence(enhanced_result.confidence)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Reasoning</span>
                    <span className="text-xs font-medium text-accent capitalize">
                      {enhanced_result.reasoning_path}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-600">Time</span>
                    <span className="text-xs text-text-700">
                      {formatTime(enhanced_result.processing_time)}s
                    </span>
                  </div>
                </div>
                <div className="border-t border-border pt-3">
                  <div className="text-sm text-text-700 prose prose-sm max-w-none">
                    <MinimalMarkdown content={enhanced_result.response.substring(0, 300) + "..."} />
                  </div>
                </div>
              </div>
            </div>

            {/* Improvement Summary */}
            <div className="border border-border rounded-12 p-4 bg-background-neutral-01">
              <h4 className="text-sm font-semibold text-text-800 mb-3">Improvement Summary</h4>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <span className="text-xs text-text-600">Confidence</span>
                  <div className={cn(
                    "text-lg font-semibold",
                    improvement_metrics.confidence_improvement >= 0 ? "text-green-600" : "text-red-600"
                  )}>
                    {formatImprovement(improvement_metrics.confidence_improvement)}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-text-600">RAG Quality</span>
                  <div className={cn(
                    "text-lg font-semibold",
                    improvement_metrics.rag_quality_improvement >= 0 ? "text-green-600" : "text-red-600"
                  )}>
                    {formatImprovement(improvement_metrics.rag_quality_improvement)}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-text-600">Processing Time</span>
                  <div className={cn(
                    "text-lg font-semibold",
                    improvement_metrics.processing_time_change <= 0 ? "text-green-600" : "text-yellow-600"
                  )}>
                    {formatImprovement(improvement_metrics.processing_time_change)}
                  </div>
                </div>
                <div>
                  <span className="text-xs text-text-600">Reasoning Upgrade</span>
                  <div className="text-lg font-semibold text-accent">
                    {improvement_metrics.reasoning_path_upgrade > 0 ? "✓ Yes" : "✗ No"}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "metrics" && (
          <div className="p-4 space-y-4">
            {/* Governance Metrics Comparison */}
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-text-800">Governance Metrics</h4>
              
              {/* Confidence */}
              <div className="border border-border rounded-12 p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-700">Confidence</span>
                  <span className={cn("text-sm font-semibold", getConfidenceColor(enhanced_result.confidence))}>
                    {formatConfidence(enhanced_result.confidence)}%
                  </span>
                </div>
                <div className="w-full bg-background-neutral-02 rounded-full h-2">
                  <div
                    className={cn(
                      "h-2 rounded-full transition-all",
                      enhanced_result.confidence >= 0.8 ? "bg-green-500" :
                      enhanced_result.confidence >= 0.6 ? "bg-yellow-500" : "bg-red-500"
                    )}
                    style={{ width: `${enhanced_result.confidence * 100}%` }}
                  />
                </div>
                <div className="text-xs text-text-600 mt-1">
                  Baseline: {formatConfidence(baseline_result.confidence)}% → Enhanced: {formatConfidence(enhanced_result.confidence)}%
                </div>
              </div>

              {/* RAG Quality */}
              <div className="border border-border rounded-12 p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-700">RAG Quality</span>
                  <span className="text-sm font-semibold text-text-800">
                    {(enhanced_result.governance_metrics.rag_quality * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-background-neutral-02 rounded-full h-2">
                  <div
                    className="bg-accent h-2 rounded-full transition-all"
                    style={{ width: `${enhanced_result.governance_metrics.rag_quality * 100}%` }}
                  />
                </div>
                <div className="text-xs text-text-600 mt-1">
                  Improvement: {formatImprovement(improvement_metrics.rag_quality_improvement)}
                </div>
              </div>

              {/* Fact Check Coverage */}
              <div className="border border-border rounded-12 p-3">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-text-700">Fact Check Coverage</span>
                  <span className="text-sm font-semibold text-text-800">
                    {(enhanced_result.governance_metrics.fact_check_coverage * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-background-neutral-02 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: `${enhanced_result.governance_metrics.fact_check_coverage * 100}%` }}
                  />
                </div>
              </div>

              {/* Reasoning Validity */}
              <div className="border border-border rounded-12 p-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-700">Reasoning Valid</span>
                  <span className={cn(
                    "text-sm font-semibold",
                    enhanced_result.governance_metrics.reasoning_valid ? "text-green-600" : "text-red-600"
                  )}>
                    {enhanced_result.governance_metrics.reasoning_valid ? "✓ Valid" : "✗ Invalid"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "details" && (
          <div className="p-4 space-y-4">
            <div>
              <h4 className="text-sm font-semibold text-text-800 mb-2">Query</h4>
              <div className="border border-border rounded-12 p-3 bg-background-neutral-01">
                <p className="text-sm text-text-700">{comparison.query}</p>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-text-800 mb-2">Full Baseline Response</h4>
              <div className="border border-border rounded-12 p-3 bg-background-neutral-01 max-h-48 overflow-y-auto">
                <div className="text-sm text-text-700 prose prose-sm max-w-none">
                  <MinimalMarkdown content={baseline_result.response} />
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-semibold text-text-800 mb-2">Full Enhanced Response</h4>
              <div className="border border-border rounded-12 p-3 bg-background-neutral-01 max-h-48 overflow-y-auto">
                <div className="text-sm text-text-700 prose prose-sm max-w-none">
                  <MinimalMarkdown content={enhanced_result.response} />
                </div>
              </div>
            </div>

            {comparison.audit_record_id && (
              <div>
                <h4 className="text-sm font-semibold text-text-800 mb-2">Audit Record</h4>
                <div className="border border-border rounded-12 p-3 bg-background-neutral-01">
                  <p className="text-xs text-text-600 font-mono">{comparison.audit_record_id}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  // Render based on position
  if (position === "bottom") {
    return (
      <div
        className={cn(
          "transition-all duration-300 ease-out",
          isOpen ? "max-h-[700px] opacity-100" : "max-h-0 opacity-0 overflow-hidden"
        )}
      >
        {panelContent}
      </div>
    );
  }

  // Side panel (for future use)
  return panelContent;
}










