import { Link } from "react-router-dom";
import { Compass } from "lucide-react";
import { Page } from "@/components/layout/Shell";
import { EmptyState } from "@/components/ui/States";
import { Button } from "@/components/ui/Button";

export default function NotFound() {
  return (
    <Page>
      <EmptyState
        icon={<Compass size={18} />}
        title="Page not found"
        description="That route doesn't exist in Reclaim."
        action={
          <Link to="/">
            <Button variant="secondary" size="sm">
              Back to overview
            </Button>
          </Link>
        }
        className="py-24"
      />
    </Page>
  );
}
