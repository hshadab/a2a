'use client';

import { CheckCircle, Circle, XCircle, Loader2, Clock } from 'lucide-react';

// A2A Task State types
export type TaskState = 'submitted' | 'working' | 'completed' | 'failed';

export interface TaskArtifact {
  id: string;
  name: string;
  mimeType: string;
  createdAt: string;
}

export interface TaskError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

export interface A2ATask {
  id: string;
  contextId?: string;
  skillId: string;
  state: TaskState;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  output?: Record<string, any>;
  artifacts?: TaskArtifact[];
  error?: TaskError;
}

interface TaskTimelineProps {
  tasks: A2ATask[];
  onTaskClick?: (task: A2ATask) => void;
}

export default function TaskTimeline({ tasks, onTaskClick }: TaskTimelineProps) {
  if (tasks.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No tasks yet
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {tasks.map((task, index) => (
        <TaskItem
          key={task.id}
          task={task}
          isLast={index === tasks.length - 1}
          onClick={onTaskClick}
        />
      ))}
    </div>
  );
}

interface TaskItemProps {
  task: A2ATask;
  isLast: boolean;
  onClick?: (task: A2ATask) => void;
}

function TaskItem({ task, isLast, onClick }: TaskItemProps) {
  const stateConfig = getStateConfig(task.state);

  return (
    <div
      className={`relative flex gap-4 ${onClick ? 'cursor-pointer hover:bg-gray-800/50' : ''} rounded-lg p-3 transition-colors`}
      onClick={() => onClick?.(task)}
    >
      {/* Timeline connector */}
      {!isLast && (
        <div className="absolute left-[1.375rem] top-12 w-0.5 h-full bg-gray-700" />
      )}

      {/* State icon */}
      <div className={`relative z-10 flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${stateConfig.bgColor}`}>
        <stateConfig.icon size={16} className={stateConfig.iconColor} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <h4 className="font-medium text-white truncate">
            {formatSkillId(task.skillId)}
          </h4>
          <span className={`px-2 py-0.5 text-xs rounded-full ${stateConfig.badgeBg} ${stateConfig.badgeText}`}>
            {task.state}
          </span>
        </div>

        <p className="text-sm text-gray-500 mt-1">
          Task ID: <code className="text-gray-400">{task.id.slice(0, 8)}...</code>
        </p>

        {/* Timestamps */}
        <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <Clock size={12} />
            {formatTimestamp(task.createdAt)}
          </span>
          {task.completedAt && (
            <span>
              Duration: {formatDuration(task.createdAt, task.completedAt)}
            </span>
          )}
        </div>

        {/* Artifacts */}
        {task.artifacts && task.artifacts.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {task.artifacts.map((artifact) => (
              <span
                key={artifact.id}
                className="px-2 py-1 text-xs bg-gray-800 text-gray-300 rounded"
              >
                {artifact.name}
              </span>
            ))}
          </div>
        )}

        {/* Error */}
        {task.error && (
          <div className="mt-2 p-2 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
            {task.error.message}
          </div>
        )}
      </div>
    </div>
  );
}

// State configuration
function getStateConfig(state: TaskState) {
  switch (state) {
    case 'submitted':
      return {
        icon: Circle,
        iconColor: 'text-gray-400',
        bgColor: 'bg-gray-800',
        badgeBg: 'bg-gray-500/20',
        badgeText: 'text-gray-400',
      };
    case 'working':
      return {
        icon: Loader2,
        iconColor: 'text-blue-400 animate-spin',
        bgColor: 'bg-blue-500/20',
        badgeBg: 'bg-blue-500/20',
        badgeText: 'text-blue-400',
      };
    case 'completed':
      return {
        icon: CheckCircle,
        iconColor: 'text-green-400',
        bgColor: 'bg-green-500/20',
        badgeBg: 'bg-green-500/20',
        badgeText: 'text-green-400',
      };
    case 'failed':
      return {
        icon: XCircle,
        iconColor: 'text-red-400',
        bgColor: 'bg-red-500/20',
        badgeBg: 'bg-red-500/20',
        badgeText: 'text-red-400',
      };
  }
}

// Format skill ID for display
function formatSkillId(skillId: string): string {
  return skillId
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Format timestamp
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

// Format duration
function formatDuration(start: string, end: string): string {
  const startDate = new Date(start);
  const endDate = new Date(end);
  const diffMs = endDate.getTime() - startDate.getTime();

  if (diffMs < 1000) {
    return `${diffMs}ms`;
  } else if (diffMs < 60000) {
    return `${(diffMs / 1000).toFixed(1)}s`;
  } else {
    const minutes = Math.floor(diffMs / 60000);
    const seconds = Math.floor((diffMs % 60000) / 1000);
    return `${minutes}m ${seconds}s`;
  }
}

// Compact timeline for sidebars
interface CompactTaskTimelineProps {
  tasks: A2ATask[];
  maxItems?: number;
}

export function CompactTaskTimeline({ tasks, maxItems = 5 }: CompactTaskTimelineProps) {
  const displayTasks = tasks.slice(0, maxItems);

  return (
    <div className="space-y-2">
      {displayTasks.map((task) => (
        <div
          key={task.id}
          className="flex items-center gap-2 text-sm"
        >
          <TaskStateIcon state={task.state} size={14} />
          <span className="text-gray-300 truncate flex-1">
            {formatSkillId(task.skillId)}
          </span>
          <span className="text-gray-500 text-xs">
            {formatTimestamp(task.createdAt)}
          </span>
        </div>
      ))}
      {tasks.length > maxItems && (
        <p className="text-xs text-gray-500 text-center">
          +{tasks.length - maxItems} more tasks
        </p>
      )}
    </div>
  );
}

// Standalone state icon
interface TaskStateIconProps {
  state: TaskState;
  size?: number;
}

export function TaskStateIcon({ state, size = 16 }: TaskStateIconProps) {
  const config = getStateConfig(state);
  const IconComponent = config.icon;
  return <IconComponent size={size} className={config.iconColor} />;
}
