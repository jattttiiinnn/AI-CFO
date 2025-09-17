from django import template

register = template.Library()

@register.filter
def in_crores(value):
    """Convert number into crores if >= 1 crore, else return as-is."""
    try:
        num = float(value)
        if num >= 1e7:  # 1 crore = 10,000,000
            return f"{num/1e7:.2f} Cr"
        else:
            # Keep it as-is but nicely formatted with commas
            return f"{num:,.2f}"
    except (ValueError, TypeError):
        return value
