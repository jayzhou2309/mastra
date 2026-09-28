import { cn } from '@/lib/utils';
import { getShortId } from '@/utils/id';

export type ItemListIdCellProps = {
  id: string;
  className?: string;
  isShortened?: boolean;
};

export function ItemListIdCell({ id, className, isShortened = true }: ItemListIdCellProps) {
  const displayId = isShortened ? getShortId(id) : id;
  return <div className={cn('truncate py-[0.6rem] text-body text-placeholder', className)}>{displayId}</div>;
}
